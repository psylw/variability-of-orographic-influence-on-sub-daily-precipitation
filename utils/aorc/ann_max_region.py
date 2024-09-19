# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr

########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
name = 'nldas'
#name = 'conus'
#%%
for window in (1,3,12,24):
    ann_max = []

    for idx,year in enumerate(range(2002,2023)):

        print(year)
        ########################## UNCOMMENT WHAT DATASET TO USE
        #dataset = '../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
        dataset = '../../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
        #dataset = '../../data/conus404/wrf2d_d01_'+str(year)+'.nc'
        ##############################################################################
        precip = xr.open_dataset(dataset)

        ########################## IF NLDAS UNCOMMENT
        precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
        ##############################################################################
        ########################## UNCOMMENT WHAT DATASET TO USE
        #precip = precip.rename({'APCP_surface': 'accum'})
        precip = precip.rename({'Rainf': 'accum'})
        #precip = precip.rename({'ACRAINLSM': 'accum'})
        ##############################################################################
        ########################## USE FOR NLDAS AND AORC
        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))
        ########################## IF CONUS UNCOMMENT
        #precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37.1,40.9))
        ##############################################################################
        precip = precip.where(precip>=0)

        size_sub_lat = int(len(precip.latitude)/4)
        size_sub_lon = int(len(precip.longitude)/4)

        ds_daily = precip.rolling(time=window).sum()
        ds_aggregated = ds_daily.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()
        ########################## IF CONUS UNCOMMENT
        #ann_max.append(ds_aggregated.max(dim='time').to_dataframe().drop(columns=['XLAT','XLONG']).rename(columns={'accum':'year'+str(year)}))
        ##############################################################################
        ann_max.append(ds_aggregated.max(dim='time').to_dataframe().rename(columns={'accum':'year'+str(year)}))

    ann_max = pd.concat(ann_max,axis=1)
    ann_max.reset_index().to_feather('../../output/'+name+'_ann_max_region'+'_window_'+str(window))

# %%
