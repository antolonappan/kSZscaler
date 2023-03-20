from database import Magneticum
import numpy as np
from astropy import units as u
import astropy.constants as c
import matplotlib.pyplot as plt

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
                        self.dataframe['m500c[Msol/h]'] / self.h)
    
    def Yksz(self) -> np.ndarray:
        first = c.sigma_T/c.c
        second = np.abs(np.array(self.dataframe['vx[km/s]'])) * (u.km/u.s)
        third = (self.Mgas * u.M_sun).to('kg') / (c.m_p)
        return (first.to('cm s') * second.to('cm/s') * third).to('Mpc^2')
    

    def Y_M(self) ->None:
        M = np.array(self.dataframe['m500c[Msol/h]']) /self.h * u.M_sun
        Y = self.Yksz()
        plt.scatter(M,Y,s=0.3)
        plt.semilogy()
        plt.semilogx()
        plt.xlabel(f'Mgas (${M.unit}$)')
        plt.ylabel(f'Yksz (${Y.unit}$)')
        plt.title('Yksz vs M500c')
        plt.show()


    

