#%%
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import xesmf as xe

#name = 'aorc'
#name = 'nldas'
#name = 'conus'
name = 'mrms'
#%%
df_nldas = xr.open_dataset('../data/NLDAS/NLDAS_FORA0125_H.A2016.nc')
df_nldas = df_nldas.rename({'lat': 'latitude', 'lon': 'longitude'})
df_aorc = xr.open_dataset('../data/aorc/larger_aorc_APCP_surface_2016.nc')
df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016.nc')

regridder_aorc = xe.Regridder(df_aorc, df_nldas, "bilinear")
regridder_conus = xe.Regridder(df_conus, df_nldas, "bilinear")

# %%
window = 1
annual_max = pd.read_feather('../output/'+name+'_ann_max_region'+'_window_'+str(window))

#%%

all_years = []
for idx,year in enumerate(range(2016,2023)):
    print(year)
    ########################## UNCOMMENT WHAT DATASET TO USE
    #dataset = '../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
    #dataset = '../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
    #dataset = '../data/conus404/wrf2d_d01_'+str(year)+'.nc'
    dataset = '../output/mrms_nldas/mrms_nldasgrid_'+str(year)+'.nc'
    ##############################################################################
    precip = xr.open_dataset(dataset).drop_vars(['step','heightAboveSea'])
    #precip = regridder_aorc(precip)
    #precip = regridder_conus(precip)
    ########################## IF NLDAS UNCOMMENT
    #precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
    ##############################################################################
    ########################## UNCOMMENT WHAT DATASET TO USE
    #precip = precip.rename({'APCP_surface': 'accum'})
    #precip = precip.rename({'Rainf': 'accum'})
    #precip = precip.rename({'ACRAINLSM': 'accum'})
    ##############################################################################
    precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

    precip = precip.where(precip>=0)*(2/60)
    precip = precip.rolling(time=30).sum()
    precip = precip.resample(time='1H').max()

    size_sub_lat = int(len(precip.latitude)/4)
    size_sub_lon = int(len(precip.longitude)/4)

    expand_lat = {}

    for i,lat in enumerate(np.sort(annual_max.latitude.unique())):
        expand_lat[lat] = precip.latitude.values.reshape(int(len(precip.latitude)/size_sub_lat), size_sub_lat)[i]

    expand_lon = {}

    for i,lon in enumerate(np.sort(annual_max.longitude.unique())):
        expand_lon[lon] = precip.longitude.values.reshape(int(len(precip.longitude)/size_sub_lon), size_sub_lon)[i]

    percent = np.arange(.1,.75,.2)
    for q in percent:
        for region in range(16):
            lat = annual_max.iloc[region].latitude
            lon = annual_max.iloc[region].longitude
            thr = annual_max.iloc[region][2::].mean()*q

            ds = precip.sel(
                                longitude = slice(np.min(expand_lon[lon]),np.max(expand_lon[lon])),latitude = slice(np.min(expand_lat[lat]),np.max(expand_lat[lat])))

            results = {}
            for lat in ds.latitude:
                for lon in ds.longitude:
                    
                    test = ds.sel(latitude=lat, longitude=lon).accum.values
                    
                    mask = test > thr
                    
                    true_indices = np.where(mask)[0]
                    
                    if len(true_indices) > 1:  
                        distances = np.diff(true_indices)
                        distances = distances[(distances!=1)&(distances<(24))]
                        if len(distances) > 0: 
                            
                            results[ (float(lat), float(lon))] = distances
                    else:
                        results[ (float(lat), float(lon))] = []  

            df = pd.DataFrame.from_dict(results, orient='index')

            df.index = pd.MultiIndex.from_tuples(df.index, names=["latitude", "longitude"])
            df['empty'] = np.nan
            df = df.apply(lambda row: row.dropna().tolist(), axis=1).reset_index(name='time_between')
            df['year'] = year
            df['percent'] = q
            df['threshold'] = thr
            df['region'] = region
            all_years.append(df)

pd.concat(all_years).reset_index(drop=True).to_feather('../output/time_between_'+name)
#%%
