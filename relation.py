from database import Magneticum
import numpy as np
from astropy import units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm
from scipy import stats
from pycachera import cache
from astropy.cosmology import WMAP7 as cosmo
from scipy.interpolate import RegularGridInterpolator
from scipy.fftpack import fftn, ifftn, fftfreq
class Scaling:
    
    def __init__(self,snap:str,box:str='',body:str='cluster') -> None:
        self.dataframe = Magneticum(snap,box,body).dataframe
        self.snap = snap
        self.box = box
        self.body = body
        self.z = Magneticum.redshift_snapshot(snap,box)
        self.h = 0.7

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
        return a*x + b
    
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
        plt.plot(M,yfit_log,label=f"$\\alpha$={popt[0]:.2f}",c='r')
        plt.legend()
    

class Distribution:

    def __init__(self,ngrid:int,snap:str,box:str='') -> None:
        

        self.boxmin, self.boxmax,  = 0., 5e5 #in kpc (500Mpc)
        self.nbins = ngrid
        self.bin_len = self.boxmax/self.nbins 
        self.h = 0.7
        self.dataframe_clu = Magneticum(snap,box,'cluster').dataframe
        self.dataframe_gal = Magneticum(snap,box,'galaxies').dataframe
        self.z = Magneticum.redshift_snapshot(snap,box)
    
    def __get_position__(self,dataframe,dtype):
        if dtype == pd.DataFrame:
            dataframe = pd.DataFrame.from_dict({'x':dataframe['x[kpc/h]']/self.h,
                                                'y':dataframe['y[kpc/h]']/self.h,
                                                'z':dataframe['z[kpc/h]']/self.h,})
        elif dtype == np.ndarray:
            dataframe = np.vstack((dataframe['x[kpc/h]']/self.h,
                                   dataframe['y[kpc/h]']/self.h,
                                   dataframe['z[kpc/h]']/self.h))
        else:
            raise ValueError('dtype must be pd.DataFrame or np.ndarray')
        
        return dataframe


    def get_postion(self,body:str) -> np.ndarray:
        if body == 'cluster' or body == 'c':
            return self.__get_position__(self.dataframe_clu,np.ndarray)
        elif body == 'galaxies' or body == 'g':
            return self.__get_position__(self.dataframe_gal,np.ndarray)
        else:
            raise ValueError('body must be cluster or galaxies')
    
    def galaxy_number_counts(self) -> np.ndarray:
        binedges = np.linspace(self.boxmin,self.boxmax,self.nbins+1)
        bincenters = 0.5*(binedges[1:]+binedges[:-1])
        ret = stats.binned_statistic_dd(self.get_postion('g').T, None, 'count', 
                                bins=[binedges, binedges, binedges])
        return ret.statistic

    def galaxy_number_density(self) -> np.ndarray:
        Ng = self.galaxy_number_counts()
        Ng_m = np.mean(Ng)
        return Ng/Ng_m - 1

    def f_growth(self) -> float:
        Om = cosmo.Om0
        Ol = 1.-Om
        return Om**(4./7.)+(1.+Om/2.)*Ol/70.   
    
    def prefactorize(self,deltag) -> float:
        f_growth = self.f_growth()
        H0 = cosmo.H0.to('/s').value
        a0 = 1./(1.+ self.z)
        bg = 5. 
        return -a0*H0*f_growth*deltag/bg 
    
    def galaxy_nd_fft(self,prefactor=True) -> np.ndarray:
        if prefactor:
            return fftn(self.prefactorize(self.galaxy_number_density()))
        else:
            return fftn(self.galaxy_number_density())
    
    def k_vectors(self) -> tuple:
        freq = fftfreq(self.nbins, d=self.bin_len*c.kpc.to('cm').value)
        kx, ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
        k_squared = kx**2 + ky**2 + kz**2
        k_squared[0, 0, 0] = 1
        return (kx,ky,kz,k_squared)
    
    def velocity_gradient(self,filter=True) -> np.ndarray:
        kx,ky,kz,k_squared = self.k_vectors()
        deltak = self.galaxy_nd_fft()
        binedges = np.linspace(self.boxmin,self.boxmax,self.nbins+1)
        bincenters = 0.5*(binedges[1:]+binedges[:-1])
        pos_c = self.get_postion('c').T
    
        R_filter = 1.5*self.bin_len*c.kpc.to('cm').value
        Wk = np.exp(-R_filter**2*k_squared/2.)
        
        if filter:
            deltak = deltak*Wk
 
        momentum_x = deltak * kx / k_squared
        momentum_y = deltak * ky / k_squared
        momentum_z = deltak * kz / k_squared

        velocity_x = np.real(ifftn(-1j*momentum_x))
        velocity_y = np.real(ifftn(-1j*momentum_y))
        velocity_z = np.real(ifftn(-1j*momentum_z))

        velocity_x_interpolator = RegularGridInterpolator((bincenters,bincenters,bincenters), 
                                                        velocity_x, bounds_error=False, 
                                                        fill_value=None, method='linear')
        velocity_y_interpolator = RegularGridInterpolator((bincenters,bincenters,bincenters), 
                                                        velocity_y, bounds_error=False, 
                                                        fill_value=None, method='linear')
        velocity_z_interpolator = RegularGridInterpolator((bincenters,bincenters,bincenters), 
                                                        velocity_z, bounds_error=False, 
                                                        fill_value=None, method='linear')


        vx = velocity_x_interpolator(pos_c)/1e5
        vy = velocity_y_interpolator(pos_c)/1e5
        vz = velocity_z_interpolator(pos_c)/1e5
        vnet = np.sqrt(vx**2+vy**2+vz**2)
        return (vx,vy,vz,vnet)


class Analysis:

    def __init__(self,grid:int,snap:str,box:str):
        self.scaling = Scaling(snap,box,'cluster')
        self.distribution = Distribution(grid,snap,box)

    def get_dataframe(self):
        df_c = self.distribution.dataframe_clu
        #df_c['Ngal'] = self.distribution.galaxy_number_density()
        vx,vy,vz,vnet = self.distribution.velocity_gradient()
        df_c['Vx'] = vx
        df_c['Vy'] = vy
        df_c['Vz'] = vz
        df_c['Vnet'] = vnet
        df_c['vnet'] = np.sqrt(df_c['vx[km/s]']**2+df_c['vy[km/s]']**2+df_c['vz[km/s]']**2)
        df_c['Mstar'] = self.scaling.Mstar.value
        df_c['Mgas'] = self.scaling.Mgas.value
        df_c['Vlos'] = self.scaling.Vlos.value
        Y,M = self.scaling.Y_M()
        df_c['Y'] = np.log(Y.value)
        df_c['M'] = np.log(M.value)
        return df_c
