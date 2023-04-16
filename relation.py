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
    
    def __init__(self,snap:str,box:str='') -> None:
        self.dataframe = Magneticum(snap,box).dataframe
        self.snap = snap
        self.box = box
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
        plt.xlabel(f'Mgas (${M.unit}$)')
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

    def __init__(self,vlos:str,grid:float,snap:str,box:str='') -> None:
        self.vlos = vlos
        self.grid = (grid * u.Mpc).to('kpc')
        self._grid_ = self.grid.value
        self.h = 0.6774
        sim = Magneticum(snap,box)
        self.dataframe = sim.dataframe
        x = np.array(self.dataframe['x[kpc/h]'])/self.h * u.kpc
        y = np.array(self.dataframe['y[kpc/h]'])/self.h * u.kpc
        z = np.array(self.dataframe['z[kpc/h]'])/self.h * u.kpc
        self.position = np.array([x,y,z])
    
    def get_axis(self) -> list:
        if self.vlos == 'x':
            sel = [1,2]
        elif self.vlos == 'y':
            sel = [0,2]
        elif self.vlos == 'z':
            sel = [0,1]
        else:
            raise ValueError('vlos must be x,y or z')
        return sel

    def get_grid_max_min(self,cube:bool=False) -> np.ndarray:
        if not cube:
            sel = self.get_axis()
            smin = min(min(self.position[sel[0]]),min(self.position[sel[1]]))
            smax = max(max(self.position[sel[0]]),max(self.position[sel[1]]))
        else:
            smin = min(min(self.position[0]),min(self.position[1]),min(self.position[2]))
            smax = max(max(self.position[0]),max(self.position[1]),max(self.position[2]))
        return np.array([smin,smax])
    
    def get_grid(self,cube:bool=False) -> np.ndarray:
        smin,smax = self.get_grid_max_min(cube)
        sdif = 1e10
        arr = []
        arr.append(smin)
        while sdif > -1:
            smin += self.grid.value
            arr.append(smin)
            sdif = smax-smin
        
        return arr
        


    def plot(self,cells:pd.DataFrame=None) -> None:
        sel = self.get_axis()
        plt.figure(figsize=(15,15))
        plt.scatter(self.position[sel[0]],self.position[sel[1]],s=1)
        if cells is not None:
            plt.scatter(cells[f'y[kpc/h]']/self.h,cells[f'z[kpc/h]']/self.h,s=4,c='k',alpha=0.6)
        g = self.get_grid()
        for i in g:
            plt.axvline(i,c='r',lw=0.5)
            plt.axhline(i,c='r',lw=0.5)

    def get_cells_i(self,i:int) -> pd.DataFrame:
        arr = self.get_grid()
        sel = self.get_axis()
        ax = ['x','y','z']
        ax_min = arr[i]
        ax_max = arr[i+1]
        return self.dataframe[(self.dataframe[f'{ax[sel[0]]}[kpc/h]']/self.h > ax_min) & (self.dataframe[f'{ax[sel[0]]}[kpc/h]']/self.h < ax_max)]
    
    def get_cube_ijk(self,i:int,j:int,k:int) -> pd.DataFrame:
        arr = self.get_grid(True)
        ax_min,ax_max = arr[i],arr[i+1]
        ay_min,ay_max = arr[j],arr[j+1]
        az_min,az_max = arr[k],arr[k+1]
        return self.dataframe[(self.dataframe['x[kpc/h]']/self.h > ax_min) & 
                              (self.dataframe['x[kpc/h]']/self.h < ax_max) &
                              (self.dataframe['y[kpc/h]']/self.h > ay_min) & 
                              (self.dataframe['y[kpc/h]']/self.h < ay_max) & 
                              (self.dataframe['z[kpc/h]']/self.h > az_min) & 
                              (self.dataframe['z[kpc/h]']/self.h < az_max)]

    def cube_data(self,i:int,j:int,k:int) -> list:
        df = self.get_cube_ijk(i,j,k)
        ind = list(df.index)
        m = np.mean(df['m500c[Msol/h]']/self.h)
        n = len(df)
        if n == 0:
            m = 0 * u.M_sun
        return [m,n,ind]
    
    @cache('/home/anto/scratch',extrarg=['_grid_'])
    def data_3d(self):
        arr = self.get_grid(True)
        ngrid = len(arr)-1
        label = np.zeros(len(self.dataframe)).astype(str)
        data = {}
        for i in tqdm(range(ngrid),desc='Calculating data',unit='cube'):
            for j in range(ngrid):
                for k in range(ngrid):
                    m,n,ind = self.cube_data(i,j,k)
                    cube = f"{i}{j}{k}"
                    data[cube] = (m,n)
                    label[ind] = cube
        self.dataframe['cube'] = label
        return data
    
    def cube_avg_quantity(self):
        data = self.data_3d()
        mass, no = [], []
        for i in tqdm(list(data.keys()),desc='Calculating density',unit='cube'):
            mass.append(data[i][0])
            no.append(data[i][1])
        mass, no = np.array(mass),np.array(no)
        mass_mask = mass == 0
        no_mask = no == 0
        masked_mass = np.ma.masked_array(mass,mass_mask)
        masked_no = np.ma.masked_array(no,no_mask)
        return masked_mass.mean(),masked_no.mean()
    
    def cube_density(self):
        mass_mean,no_mean = self.cube_avg_quantity()
        data = self.data_3d()
        mass_dens = np.zeros(len(self.dataframe))
        no_dens = np.zeros(len(self.dataframe))
        cube = list(self.dataframe['cube'])
        for ind, key  in tqdm(enumerate(cube),desc='Calculating density',unit='cube'):
            print(key)
            mass_dens[ind] = data[str(key)][0]/mass_mean
            no_dens[ind] = data[str(key)][1]/no_mean
        self.dataframe['mass_dens'] = mass_dens
        self.dataframe['no_dens'] = no_dens
            


    
    def get_cell_ij(self,i:int,j:int) -> pd.DataFrame:
        arr = self.get_grid()
        sel = self.get_axis()
        ax = ['x','y','z']
        ax_min = arr[j]
        ax_max = arr[j+1]
        df = self.get_cells_i(i)
        return df[(df[f'{ax[sel[1]]}[kpc/h]']/self.h > ax_min) & (df[f'{ax[sel[1]]}[kpc/h]']/self.h < ax_max)]
    
    def cell_avg_mass(self,i:int,j:int) -> u.M_sun:
        df = self.get_cell_ij(i,j)
        m = df['m500c[Msol/h]']/self.h
        if len(m) == 0:
            return 0*u.M_sun
        return np.mean(m)*u.M_sun
    
    @cache('/home/anto/scratch')
    def cells_avg_mass(self) -> u.M_sun:
        arr = self.get_grid()
        arr = arr[:-1]
        mass = []
        for i in tqdm(range(len(arr)),desc='Calculating mass of cells in y direction rows',unit='row',position=0):
            for j in tqdm(range(len(arr)),desc='Calculating mass of cells in z direction columns',unit='cell',position=1,leave=False):
                mass.append(self.cell_avg_mass(i,j).value)
        return np.mean(mass)*u.M_sun
    
    def cell_number(self,i:int,j:int) -> int:
        df = self.get_cell_ij(i,j)
        return len(df)
    
    @cache('/home/anto/scratch')
    def cells_number(self) -> int:
        arr = self.get_grid()
        arr = arr[:-1]
        number = []
        for i in tqdm(range(len(arr)),desc='Calculating number of cells in y direction rows',unit='row',position=0):
            for j in tqdm(range(len(arr)),desc='Calculating number of cells in z direction columns',unit='cell',position=1,leave=False):
                number.append(self.cell_number(i,j))
        return np.mean(number)
    
    def cell_mass_density(self,i:int,j:int) -> float:
        m_i = self.cell_avg_mass(i,j)
        m = self.cells_avg_mass()
        return m_i/m 
    
    def cell_number_density(self,i:int,j:int) -> float:
        n_i = self.cell_number(i,j)
        n = self.cells_number()
        return n_i/n
    
    @cache('/home/anto/scratch')
    def get_density(self) -> tuple:
        mass_den = np.zeros(len(self.dataframe))
        numb_den = np.zeros(len(self.dataframe))
        arr = self.get_grid()
        arr = arr[:-1]
        for i in tqdm(range(len(arr)),desc='Calculating mass of cells in y direction rows',unit='row',position=0):
            for j in tqdm(range(len(arr)),desc='Calculating mass of cells in z direction columns',unit='cell',position=1,leave=False):
                _df_ = self.get_cell_ij(i,j)
                mass_den[list(_df_.index)] = np.ones(len(_df_))*self.cell_mass_density(i,j)
                numb_den[list(_df_.index)] = np.ones(len(_df_))*self.cell_number_density(i,j)
        return (mass_den,numb_den)
    
    def add_density(self) -> None:
        mass_den,numb_den = self.get_density()
        self.dataframe['mass_density'] = mass_den
        self.dataframe['number_density'] = numb_den
        print('Density data added to dataframe')
    

    








    

