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
                        self.dataframe['m500c[Msol/h]'] * self.h)
    
    def Yksz(self) -> np.ndarray:
        return (c.sigma_T/c.c) * (self.dataframe['vx[km/s]'] *1e3) * (self.Mgas/(c.m_p))

    def Y_M(self) ->None:
        plt.scatter(np.array(self.dataframe['m500c[Msol/h]']),self.Yksz(),s=0.3)
        plt.xlabel('Mgas')
        plt.ylabel('Yksz')
        plt.title('Yksz vs M500c')
        plt.show()


    

