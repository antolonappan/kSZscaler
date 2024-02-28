#https://arxiv.org/pdf/2201.01305.pdf


from database import Magneticum
import numpy as np
from astropy import units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import modin.pandas as pd
from tqdm import tqdm
from scipy import stats
from pycachera import cache
from astropy.cosmology import WMAP7 as cosmo
from scipy.interpolate import RegularGridInterpolator
from scipy.fftpack import fftn, ifftn, fftfreq
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


class Scaling:
    
    def __init__(self,snap:str,box:str='',body:str='cluster') -> None:
        self.dataframe = Magneticum(snap,box,body).dataframe
        self.snap = snap
        self.box = box
        self.body = body
        self.z = Magneticum.redshift_snapshot(snap,box)
        self.h = 0.704

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
    def Temp(self) -> np.ndarray:
        return np.array(self.dataframe['T[kev]']) * u.keV
    
    @property
    def Vlos(self) -> np.ndarray:
        return np.array(self.dataframe['vz[km/s]']) * (u.km/u.s)
    
    def Yksz(self) -> np.ndarray:
        first = c.sigma_T/c.c
        second = np.abs(self.Vlos)
        third = self.Mgas.to('kg') / (c.m_p)
        return (first.to('cm s') * second.to('cm/s') * third).to('Mpc^2')

    def Ytsz(self) -> np.ndarray:
        first = (c.sigma_T*c.k_B*self.Temp.to('K',equivalencies=u.temperature_energy()))/(c.m_e*c.c**2)
        second = self.Mgas.to('kg') / (c.m_p)
        return (first * second).to('Mpc^2')

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

class photoz:
    def __init__(self,z):
        self.z = z
        self.sigma_z = 0.01*np.exp(self.z/2.5)
        c = 299792.458 
        H0 = 71
        H0_kpc = H0 / 1000
        dD_dz = c / H0_kpc
        self.sigma_D = abs(dD_dz * self.sigma_z) * u.kpc
    

class Distribution:

    def __init__(self,ngrid:int,snap:str,box:str='',zerr:float=0.0) -> None:
        self.boxsize = (352/0.704)*1000
        
        self.boxmin, self.boxmax,  = 0., 5e5 #in kpc (500Mpc)
        self.nbins = ngrid
        self.bin_len = self.boxmax/self.nbins 
        self.h = 0.704
        self.dataframe_clu = Magneticum(snap,box,'cluster').dataframe
        self.dataframe_gal = Magneticum(snap,box,'galaxies').dataframe
        self.z = Magneticum.redshift_snapshot(snap,box)
        self.zerr = zerr

    def __add_zerr__(self) -> np.ndarray:
        z = np.array(self.dataframe_gal['z[kpc/h]'])/self.h
        z_with_err = np.random.normal(z,self.zerr)
        z_with_err[z_with_err<0] = z_with_err[z_with_err<0]+self.boxsize
        z_with_err[z_with_err>self.boxsize] = z_with_err[z_with_err>self.boxsize]-self.boxsize
        return z_with_err

    
    def __get_position__(self,dataframe,dtype,add_zerr=False) -> np.ndarray:
        if add_zerr:
            z = self.__add_zerr__()
        if dtype == pd.DataFrame:
            dataframe = pd.DataFrame.from_dict({'x':dataframe['x[kpc/h]']/self.h,
                                                'y':dataframe['y[kpc/h]']/self.h,
                                                'z':dataframe['z[kpc/h]']/self.h if not add_zerr else z})
        elif dtype == np.ndarray:
            dataframe = np.vstack((dataframe['x[kpc/h]']/self.h,
                                   dataframe['y[kpc/h]']/self.h,
                                   dataframe['z[kpc/h]']/self.h if not add_zerr else z))
        else:
            raise ValueError('dtype must be pd.DataFrame or np.ndarray')
        
        return dataframe


    def get_postion(self,body:str) -> np.ndarray:
        if body == 'cluster' or body == 'c':
            return self.__get_position__(self.dataframe_clu,np.ndarray)
        elif body == 'galaxies' or body == 'g':
            return self.__get_position__(self.dataframe_gal,np.ndarray,True)
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
    
        R_filter = 2*self.bin_len*c.kpc.to('cm').value
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
        
        
        vx = velocity_x_interpolator(pos_c)
        vy = velocity_y_interpolator(pos_c)
        vz = velocity_z_interpolator(pos_c)
        vnet = np.sqrt(vx**2+vy**2+vz**2)
        return (vx,vy,vz,vnet)

class Analysis:

    def __init__(self,grid:int,snap:str,box:str,zerr:float=0.0) -> None:
        self.scaling = Scaling(snap,box,'cluster')
        self.distribution = Distribution(grid,snap,box,zerr)

    def get_dataframe(self):
        df_c = self.distribution.dataframe_clu
        #df_c['Ngal'] = self.distribution.galaxy_number_density()
        vx,vy,vz,vnet = self.distribution.velocity_gradient()
        df_c['Vx'] = vx
        df_c['Vy'] = vy
        df_c['Vz'] = vz
        df_c['Vnet'] = vnet
        df_c['vnet'] = np.sqrt(df_c['vx[km/s]']**2+df_c['vy[km/s]']**2+df_c['vz[km/s]']**2)
        df_c['vz'] = df_c['vz[km/s]']
        df_c['Mstar'] = self.scaling.Mstar.value
        df_c['Mgas'] = self.scaling.Mgas.value
        df_c['Vlos'] = self.scaling.Vlos.value
        Y,M = self.scaling.Y_M()
        df_c['Yksz'] = np.log(Y.value)
        df_c['M'] = np.log(M.value)
        df_c['Ytsz'] = np.log(self.scaling.Ytsz().value)
        return df_c




class RandomForest:
    def __init__(self, data_df):
        self.data_df = data_df
        self.add_quantile()
        self.X = data_df[['Vz','Mstar', 'M','Vlos','M_q']]
        self.y = data_df['Yksz']
        self.best_hyperparameters = {}
        self.models = {}
        self.r2_scores = {}
        self.rmse_scores = {}
        self.cvmodels = {}
        self.cv_scores = {}
    
    def add_quantile(self,q_div=4):
        Q = self.get_q(q_div)
        labels = list(Q.keys())
        q_list = [0,0.25,0.5,0.75,1]
        self.data_df['M_q'] = pd.qcut(self.data_df['M'],q=q_list,labels=labels)
    
    
    def get_q(self,num_divisions,str=False):
        q_dict = {}
        step = 1 / num_divisions

        for i in range(num_divisions):
            start = i * step
            end = start + step
            key = f'Q{i + 1}'
            if str:
                q_dict[key] = f'{np.round(start,2)}-{np.round(end,2)}'
            else:
                q_dict[key] = [np.round(start,2), np.round(end,2)]

        return q_dict

    def split_data(self, test_size=0.8, random_state=42):
        
        XX_train, XX_test, yy_train, yy_test = train_test_split(self.X, self.y, test_size=test_size,stratify=self.X['M_q'],random_state=random_state)
        
        self.X_train, self.X_test  = XX_train, XX_test
        self.y_train, self.y_test  = yy_train, yy_test

    def get_data(self,which,q_div=None,quantile=None):
        if which == 'train':
            X, y = self.X_train, self.y_train
        elif which == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("Invalid value for 'which'. Use 'train', 'test', or 'tune'.")
        
        if quantile is not None:
            assert q_div is not None, 'q_div is not given'
            Q = self.get_q(q_div)[quantile]
            print(f'Using {Q} quantile for Mass cut')
            m_mask = (X['M'] >= X['M'].quantile(Q[0])) & (X['M'] < X['M'].quantile(Q[1]))
            X = X.loc[m_mask]
            y = y.loc[m_mask]
        return X,y

    def get_fit(self, which, zscore=False, q_div=4,quantile=None, return_xy=False):
        
        if which == 'train':
            X, y = self.X_train, self.y_train
        elif which == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("Invalid value for 'which'. Use 'train', 'test', or 'tune'.")
        
        X2, y2 = self.get_data(which,q_div=q_div,quantile=quantile)


        # Perform linear fit using numpy.polyfit
        fit = np.polyfit(X2['M'], y2, 1)
        slope, intercept = fit

            
        if zscore:
            if which != 'train':
                slope, intercept = self.get_fit('train')
            z_scores = (y - (slope * X['M'] + intercept)) / np.std(y)
            #outliers_mask = z_scores > 3  # Define a threshold for outliers (e.g., Z-score > 3)
            #X_cleaned = X.loc[~outliers_mask]
            #y_cleaned = y.loc[~outliers_mask]
            #return np.polyfit(X_cleaned['M'], y_cleaned, 1)
            return z_scores
        
        if return_xy:
            return X2['M'], y2, X['M'], slope * X['M'] + intercept, slope

        return fit
    
    def plot_fit(self, which='train',q_div=4,quantile=None):

        if (q_div is not None) and (quantile is None):
            Q = self.get_q(q_div,str=True)
            for i in range(q_div):
                q = f'Q{i + 1}'
                x_q, y_q, x_fit_q, y_fit_q,slope = self.get_fit(which, return_xy=True, q_div=q_div,quantile=q)
                plt.scatter(x_q, y_q, s=1, alpha=0.5)
                plt.plot(x_fit_q, y_fit_q,label=f'{Q[q]} quantile, slope={slope:.2f}')
        elif (q_div is None) and (quantile is None):
            x,y,x_fit, y_fit,slope = self.get_fit(which, return_xy=True, quantile=quantile)
            plt.scatter(x, y, s=1, alpha=0.5)
            plt.plot(x_fit, y_fit, color='red',label=f'{quantile} quantile, slope={slope:.2f}')
        elif (q_div is not None) and (quantile is not None):
            x,y,x_fit, y_fit,slope = self.get_fit(which, return_xy=True, q_div=q_div,quantile=quantile)
            plt.scatter(x, y, s=1, alpha=0.5)
            plt.plot(x_fit, y_fit, color='red',label=f'{quantile} quantile, slope={slope:.2f}')
    
        plt.xlabel('M')
        plt.ylabel('Yksz')
        plt.legend()
    
    def clean_data(self, zscore_threshold=3):
        z_scores = np.abs(self.get_fit('train', zscore=True))
        outliers_mask = z_scores > zscore_threshold
        self.X_train = self.X_train.loc[~outliers_mask]
        self.y_train = self.y_train.loc[~outliers_mask]
        z_scores = np.abs(self.get_fit('test', zscore=True))
        outliers_mask = z_scores > zscore_threshold
        self.X_test = self.X_test.loc[~outliers_mask]
        self.y_test = self.y_test.loc[~outliers_mask]


    def find_best_hyperparameters(self, param_grid, cv=5):
        X_train = self.X_train.drop(columns=['M_q','Vlos'])
        rf_regressor = RandomForestRegressor()
        grid_search = GridSearchCV(rf_regressor, param_grid, cv=cv, n_jobs=4, verbose=2)
        grid_search.fit(X_train, self.y_train)
        self.best_hyperparameters = grid_search.best_params_

    def fitted_model(self, features,):
        rf_regressor = RandomForestRegressor(**self.best_hyperparameters)
        X_train, y_train = self.get_data('train')
        X_train = X_train[features]
        rf_regressor.fit(X_train, y_train,sample_weight=np.exp(X_train['M']))
        return rf_regressor

    def fit_with_hyperparameters(self, features, target,regressor=None):

        rf_regressor = RandomForestRegressor(**self.best_hyperparameters)
        X_train, y_train = self.get_data('train')
        X_train = X_train[features]
        rf_regressor.fit(X_train, y_train,sample_weight=np.exp(X_train['M']))
        features_key = "_".join(features) # Convert list of features to a string key
        self.models[features_key] = rf_regressor

        y_train_pred = rf_regressor.predict(X_train.values.reshape(-1, len(features)))
        self.r2_scores[features_key] = {'train_r2': r2_score(y_train, y_train_pred)}
        self.rmse_scores[features_key] = {'train_rmse': RMSE(y_train, y_train_pred)}


        X_test, y_test = self.get_data('test')
        X_test = X_test[features]
        y_test_pred = rf_regressor.predict(X_test.values.reshape(-1, len(features)))
        self.r2_scores[features_key]['test_r2'] = r2_score(y_test, y_test_pred)
        self.rmse_scores[features_key]['test_rmse'] = RMSE(y_test, y_test_pred)

    def calculate_scatter(self, features):
        features_key = "_".join(features) 
        rf_regressor = self.models[features_key]
        X_train, y_train = self.get_data('train')
        X_test, y_test = self.get_data('test')
        X_train = X_train[features]
        X_test = X_test[features]
        # Convert features to a 2D array
        X_train = X_train.values.reshape(-1, len(features))
        X_test = X_test.values.reshape(-1, len(features))
        y_train_pred = rf_regressor.predict(X_train)
        y_test_pred = rf_regressor.predict(X_test)
        train_scatter = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_scatter = np.sqrt(mean_squared_error(y_test, y_test_pred))
        return train_scatter, test_scatter

    def plot_scatter_statistics(self):

        train_scatters = []
        test_scatters = []
        feature_sets = list(self.models.keys())
        feature_sets = [feature_set.split("_") for feature_set in feature_sets]

        for features in feature_sets:
            train_scatter, test_scatter = self.calculate_scatter(features)
            train_scatters.append(train_scatter)
            test_scatters.append(test_scatter)
        
        train_percent_change = [100 * (sc - train_scatters[0]) / train_scatters[0] for sc in train_scatters]
        test_percent_change = [100 * (sc - test_scatters[0]) / test_scatters[0] for sc in test_scatters]


        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(feature_sets)), train_scatters, width=0.4, align='center', label='Train Scatter')
        plt.bar(np.arange(len(feature_sets)) + 0.4, test_scatters, width=0.4, align='center', label='Test Scatter')
        plt.xticks(np.arange(len(feature_sets)), feature_sets, rotation=45)
        plt.xlabel('Feature Set')
        plt.ylabel('Scatter')
        plt.title('Scatter for Different Feature Sets')
        plt.legend()

        for i, feature_set in enumerate(feature_sets):
            plt.annotate(f'{train_percent_change[i]:.2f}%', xy=(i, train_scatters[i]), xytext=(-10, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            plt.annotate(f'{test_percent_change[i]:.2f}%', xy=(i + 0.4, test_scatters[i]), xytext=(-10, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8)


        plt.tight_layout()
        plt.show()

    def plot_r2_statistics(self):

        feature_sets = list(self.models.keys())
        feature_sets_name = [feature_set.split("_") for feature_set in feature_sets]
        train_r2_values = [self.r2_scores[features]['train_r2'] for features in feature_sets]
        test_r2_values = [self.r2_scores[features]['test_r2'] for features in feature_sets]

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(feature_sets)), train_r2_values, width=0.4, align='center', label='Train R2')
        plt.bar(np.arange(len(feature_sets)) + 0.4, test_r2_values, width=0.4, align='center', label='Test R2')
        plt.xticks(np.arange(len(feature_sets)), feature_sets_name, rotation=45)
        plt.xlabel('Feature Set')
        plt.ylabel('R2 Score')
        plt.title('R2 Score for Different Feature Sets')
        plt.legend()

        for i, train_r2 in enumerate(train_r2_values):
            plt.text(i, train_r2 + 0.01, f'{train_r2:.2f}', ha='center', va='bottom')
        for i, test_r2 in enumerate(test_r2_values):
            plt.text(i + 0.4, test_r2 + 0.01, f'{test_r2:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
    
    def plot_RMSE(self):
        feature_sets = list(self.models.keys())
        feature_sets_name = [feature_set.split("_") for feature_set in feature_sets]
        train_rmse_values = [self.rmse_scores[features]['train_rmse'] for features in feature_sets]
        test_rmse_values = [self.rmse_scores[features]['test_rmse'] for features in feature_sets]

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(feature_sets)), train_rmse_values, width=0.4, align='center', label='Train RMSE')
        plt.bar(np.arange(len(feature_sets)) + 0.4, test_rmse_values, width=0.4, align='center', label='Test RMSE')
        plt.xticks(np.arange(len(feature_sets)), feature_sets_name, rotation=45)
        plt.xlabel('Feature Set')
        plt.ylabel('RMSE')
        plt.title('RMSE for Different Feature Sets')
        plt.legend()

        for i, train_rmse in enumerate(train_rmse_values):
            plt.text(i, train_rmse + 0.01, f'{train_rmse:.2f}', ha='center', va='bottom')
        for i, test_rmse in enumerate(test_rmse_values):
            plt.text(i + 0.4, test_rmse + 0.01, f'{test_rmse:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
    def fit_with_hyperparameters_cv(self, features, target, cv=5):
        rf_regressor = RandomForestRegressor(**self.best_hyperparameters)
        X_train = self.X_train[features]

        # Using cross_val_score to perform cross-validation
        cv_scores = cross_val_score(rf_regressor, X_train, self.y_train, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)  # Convert negative MSE scores to RMSE

        features_key = "_".join(features)
        self.cvmodels[features_key] = rf_regressor

        # Store cross-validated RMSE scores
        self.cv_scores[features_key] = {
            'cv_rmse_mean': np.mean(rmse_scores),
            'cv_rmse_std': np.std(rmse_scores)
        }

    def plot_cv_rmse(self):
        feature_sets = list(self.cvmodels.keys())
        feature_sets_name = [feature_set.split("_") for feature_set in feature_sets]
        cv_rmse_means = [self.cv_scores[features]['cv_rmse_mean'] for features in feature_sets]
        cv_rmse_stds = [self.cv_scores[features]['cv_rmse_std'] for features in feature_sets]

        plt.figure(figsize=(10, 6))
        plt.errorbar(np.arange(len(feature_sets)), cv_rmse_means, yerr=cv_rmse_stds, fmt='o', capsize=5)
        plt.xticks(np.arange(len(feature_sets)), feature_sets_name, rotation=45)
        plt.xlabel('Feature Set')
        plt.ylabel('CV RMSE')
        plt.title('Cross-Validated RMSE for Different Feature Sets')
        plt.tight_layout()
        plt.show()


def RMSE(true, predict):
    return np.sqrt(((true-predict)**2.).sum()/len(true))