# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr
import xesmf as xe
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
#name = 'nldas'
#name = 'conus'
name = 'conus_all'
#name = 'mrms'
#%%
df_nldas = xr.open_dataset('../../data/NLDAS/NLDAS_FORA0125_H.A2016.nc')
df_nldas = df_nldas.rename({'lat': 'latitude', 'lon': 'longitude'})
df_aorc = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_2016.nc')
df_conus = xr.open_dataset('../../data/conus404/wrf2d_d01_2016.nc')

regridder_aorc = xe.Regridder(df_aorc, df_nldas, "bilinear")
regridder_conus = xe.Regridder(df_conus, df_nldas, "bilinear")
#%%
#for window in (1,3,12,24):
window = 1
#for window in (1*30,3*30,12*30,24*30):
ann_max = []

#for idx,year in enumerate(range(2016,2023)):
for idx,year in enumerate(range(1980,2023)):

    print(year)
    ########################## UNCOMMENT WHAT DATASET TO USE
    #dataset = '../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
    #dataset = '../../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
    dataset = '../../data/conus404/wrf2d_d01_'+str(year)+'.nc'
    #dataset = '../../output/mrms_nldas/mrms_nldasgrid_'+str(year)+'.nc'
    ##############################################################################
    #precip = xr.open_dataset(dataset).drop_vars(['step','heightAboveSea'])
    precip = xr.open_dataset(dataset)
    #precip = regridder_aorc(precip)
    #precip = regridder_conus(precip)

    ########################## IF NLDAS UNCOMMENT
    #precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
    ##############################################################################
    ########################## UNCOMMENT WHAT DATASET TO USE
    #precip = precip.rename({'APCP_surface': 'accum'})
    #precip = precip.rename({'Rainf': 'accum'})
    precip = precip.rename({'ACRAINLSM': 'accum'})
    ##############################################################################
    precip = precip.sel(longitude = slice(-109.04,-103.96),latitude = slice(36.98,41.02))
    #precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

    precip = precip.where(precip>=0)
    #precip = precip.where(precip>=0)*(2/60)

    size_sub_lat = int(len(precip.latitude)/4)
    size_sub_lon = int(len(precip.longitude)/4)

    ds_daily = precip.rolling(time=window).sum()
    ds_aggregated = ds_daily.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()

    ann_max.append(ds_aggregated.max(dim='time').to_dataframe().rename(columns={'accum':'year'+str(year)}))

#window = int((window*2)/60)
ann_max = pd.concat(ann_max,axis=1)
ann_max.reset_index().to_feather('../../output/'+name+'_ann_max_region'+'_window_'+str(window))

# %%
