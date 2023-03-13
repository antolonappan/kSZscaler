import numpy as np
import wget
import shutil
import os
import gzip




class Magneticum:

    def __init__(self,snap,box=''):
        self.snap = snap
        self.box = box
        self.fname = self.__datafile__(f'{self.snap}{self.box}.txt' )
        self.link = f"http://wwwmpa.mpa-garching.mpg.de/HydroSims/Magneticum/Downloads/Magneticum/Box2{box}_hr/snap_{int(snap)}/cluster.txt.gz"
    
    def __datafile__(self,fname):
        dir_base = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_base,'Data',fname)


    
    def download_data(self):
        print(f'Downloading {self.link}')
        wget.download(self.link, out=self.fname+'.gz')
        
    
    def unzip_data(self):
        print(f'Unzipping {self.fname}.gz')
        with gzip.open(self.fname+'.gz', 'rb') as s_file, \
                open(self.fname, 'wb') as d_file:
            shutil.copyfileobj(s_file, d_file, 65536)
    
    def get_data(self):
        self.download_data()
        self.unzip_data()
        os.remove(self.fname+'.gz')
        print(f'Data saved to {self.fname}')
    
        






#http://wwwmpa.mpa-garching.mpg.de/HydroSims/Magneticum/Downloads/Magneticum/Box2_hr/snap_140/cluster.txt.gz
#http://wwwmpa.mpa-garching.mpg.de/HydroSims/Magneticum/Downloads/Magneticum/Box2b_hr/snap_031/cluster.txt.gz