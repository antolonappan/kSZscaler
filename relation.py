from database import Magneticum
import numpy as np
from astropy import units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm
from pycachera import cache
class Scaling:
    
    def __init__(self,snap:str,box:str='',body:str='cluster') -> None:
        self.dataframe = Magneticum(snap,box,body).dataframe
        self.snap = snap
        self.box = box
        self.body = body
        self.z = Magneticum.redshift_snapshot(snap,box)
        self.h = 0.6774

    @property
    def Mgas(self) -> np.ndarray:
        return np.array(self.dataframe['gas_frac'].values * \
                        self.dataframe['m500c[Msol/h]'] / self.h) * u.M_sun
    
    @property
    def Mstar(self) -> np.ndarray:
        return np.array(self.dataframe['star_frac'].values * \
                        self.dataframe['m500c[Msol/h]'] / self.h) * u.M_sun
    @property
    def Mhalo(self) -> np.ndarray:
        return np.array(self.dataframe['m500c[Msol/h]'] / self.h) * u.M_sun
    
    @property
    def Vlos(self) -> np.ndarray:
        return np.abs(np.array(self.dataframe['vx[km/s]'])) * (u.km/u.s)
    
    def Yksz(self) -> np.ndarray:
        first = c.sigma_T/c.c
        second = self.Vlos
        third = self.Mgas.to('kg') / (c.m_p)
        return (first.to('cm s') * second.to('cm/s') * third).to('Mpc^2')

    def gas_halo_relation(self) -> None:
        Mhalo = np.array(self.dataframe['m500c[Msol/h]']) /self.h * u.M_sun
        Mgas = self.Mgas * u.M_sun
        plt.scatter(Mgas,Mhalo,s=0.3)
        plt.semilogy()
        plt.semilogx()
        plt.xlabel(f'Mgas (${Mgas.unit}$)')
        plt.ylabel(f'Mhalo (${Mhalo.unit}$)')
        plt.title('Mgas vs Mhalo')
        plt.show()
    
    def velo_halo_relation(self,ax) -> None:
        Mhalo = np.array(self.dataframe['m500c[Msol/h]']) /self.h * u.M_sun
        Vgas = np.abs(np.array(self.dataframe[f'v{ax}[km/s]'])) * (u.km/u.s)
        plt.scatter(Vgas,Mhalo,s=0.3)
        plt.semilogy()
        #plt.semilogx()
        plt.xlabel(f'Vgas (${Vgas.unit}$)')
        plt.ylabel(f'Mhalo (${Mhalo.unit}$)')
        plt.title('Vgas vs Mhalo')
        plt.show()
    

    def Y_M(self) -> tuple:
        M = np.array(self.dataframe['m500c[Msol/h]']) /self.h * u.M_sun
        Y = self.Yksz()
        return (Y,M)
    
    def plot_Y_M(self) -> None:
        Y,M = self.Y_M()
        plt.scatter(M,Y,s=0.3)
        plt.semilogy()
        plt.semilogx()
        plt.xlabel(f'M500c (${M.unit}$)')
        plt.ylabel(f'Yksz (${Y.unit}$)')
        plt.title('Yksz vs M500c')
        plt.show()

    def power_law(self,x,a,b):
        return a*x**b
    
    def Y_M_fit(self) -> tuple:
        Y,M = self.Y_M()
        xdata_log = np.log(M.value)
        ydata_log = np.log(Y.value)
        popt, pcov = curve_fit(self.power_law, xdata_log, ydata_log)
        yfit_log = self.power_law(xdata_log,*popt)
        return (popt,pcov,np.exp(yfit_log))
    
    def plot_Y_M_fit(self) -> None:
        Y,M = self.Y_M()
        popt,pcov,yfit_log = self.Y_M_fit()
        plt.scatter(M,Y,s=0.3)
        plt.semilogy()
        plt.semilogx()
        plt.xlabel(f'Mgas (${M.unit}$)')
        plt.ylabel(f'Yksz (${Y.unit}$)')
        plt.title('Yksz vs M500c')
        plt.plot(M,yfit_log,label=f"$\\alpha$={popt[1]:.2f}",c='r')
        plt.legend()
    

class Distribution:

    def __init__(self,grid:float,snap:str,box:str='') -> None:
        self.grid = (grid * u.Mpc).to('kpc')
        self._grid_ = self.grid.value
        self.h = 0.7
        self.dataframe_clu = Magneticum(snap,box,'cluster').dataframe
        self.dataframe_gal = Magneticum(snap,box,'galaxies').dataframe
        self.dataframe = self.calculate_number_density()
    
    def __modify_dataframe_cluster__(self):
        dataframe = pd.DataFrame.from_dict({'x':self.dataframe_clu['x[kpc/h]']/self.h,
                                            'y':self.dataframe_clu['y[kpc/h]']/self.h,
                                            'z':self.dataframe_clu['z[kpc/h]']/self.h,})
        dataframe['m500c[Msol]'] = self.dataframe_clu['m500c[Msol/h]']/self.h
        dataframe['Veff'] = np.sqrt(self.dataframe_clu['vx[km/s]']**2 + self.dataframe_clu['vy[km/s]']**2 + self.dataframe_clu['vz[km/s]']**2)
        return dataframe

    def __modify_dataframe_galaxies__(self):
        dataframe = pd.DataFrame.from_dict({'x':self.dataframe_gal['x[kpc/h]']/self.h,
                                            'y':self.dataframe_gal['y[kpc/h]']/self.h,
                                            'z':self.dataframe_gal['z[kpc/h]']/self.h,})
        return dataframe

    
    def calculate_number_density(self):
        data = self.__modify_dataframe_cluster__()
        data1 = self.__modify_dataframe_galaxies__()


        cube_size = self.grid.value
        # Calculate the minimum and maximum coordinates for each axis in the cluster data
        min_x, max_x = data['x'].min(), data['x'].max()
        min_y, max_y = data['y'].min(), data['y'].max()
        min_z, max_z = data['z'].min(), data['z'].max()

        # Calculate the number of cubes along each axis
        num_cubes_x = int(np.ceil((max_x - min_x) / cube_size))
        num_cubes_y = int(np.ceil((max_y - min_y) / cube_size))
        num_cubes_z = int(np.ceil((max_z - min_z) / cube_size))

        # Function to calculate cube indices and unique cube ID
        def assign_cube_ids(df):
            df['cube_x'] = np.floor((df['x'] - min_x) / cube_size).astype(int)
            df['cube_y'] = np.floor((df['y'] - min_y) / cube_size).astype(int)
            df['cube_z'] = np.floor((df['z'] - min_z) / cube_size).astype(int)
            df['cube_id'] = (df['cube_x'] * num_cubes_y * num_cubes_z +
                            df['cube_y'] * num_cubes_z +
                            df['cube_z'])
            return df

        # Assign cube IDs to both galaxy clusters and galaxy data
        data = assign_cube_ids(data)
        data1 = assign_cube_ids(data1)

        # Calculate the number of galaxy clusters in each cube
        clusters_per_cube = data.groupby('cube_id').size().reset_index(name='num_clusters')

        # Calculate the mean number of galaxy clusters in all cubes
        mean_num_clusters = clusters_per_cube['num_clusters'].mean()

        data = data.merge(clusters_per_cube[['cube_id', 'num_clusters']], on='cube_id', how='left')

        # Calculate the number density of galaxy clusters in each cube
        clusters_per_cube['cluster_number_density'] = clusters_per_cube['num_clusters'] / mean_num_clusters

        # Associate the number density with each galaxy cluster in the data
        data = data.merge(clusters_per_cube[['cube_id', 'cluster_number_density']], on='cube_id', how='left')

        # Calculate the number of galaxies in each cube
        galaxies_per_cube = data1.groupby('cube_id').size().reset_index(name='num_galaxies')

        data = data.merge(galaxies_per_cube[['cube_id', 'num_galaxies']], on='cube_id', how='left')

        # Calculate the mean number of galaxies in all cubes
        mean_num_galaxies = galaxies_per_cube['num_galaxies'].mean()

        # Calculate the number density of galaxies in each cube
        galaxies_per_cube['galaxy_number_density'] = galaxies_per_cube['num_galaxies'] / mean_num_galaxies

        # Associate the number density with each galaxy cluster in the data
        data = data.merge(galaxies_per_cube[['cube_id', 'galaxy_number_density']], on='cube_id', how='left')

        data = data.drop(['cube_x', 'cube_y', 'cube_z', 'cube_id'], axis=1)

        return data
        

        

        








        

