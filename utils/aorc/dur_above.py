# get the duration above annual max intensity 
#%%
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import rasterio
import xesmf as xe
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
#name = 'nldas'
#name = 'conus'
name = 'conus_all'
#%%
df_nldas = xr.open_dataset('../../data/NLDAS/NLDAS_FORA0125_H.A2016.nc')
df_nldas = df_nldas.rename({'lat': 'latitude', 'lon': 'longitude'})
df_aorc = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_2016.nc')
df_conus = xr.open_dataset('../../data/conus404/wrf2d_d01_2016.nc')

regridder_aorc = xe.Regridder(df_aorc, df_nldas, "bilinear")
regridder_conus = xe.Regridder(df_conus, df_nldas, "bilinear")
# %%
#for window in (1,3,12,24):
window = 1
#storms = pd.read_feather('../../output/'+name+'_storms_above_thr'+'_window_'+str(window))
storms = pd.read_feather('../../output/'+name+'_lower_storms_above_thr'+'_window_'+str(window))
storms = storms.drop(columns='index')
storms['year'] = storms.time.dt.year

annual_max = pd.read_feather('../../output/'+name+'_ann_max_region'+'_window_'+str(window))
for idx,year in enumerate(range(1980,2023)):
#for year in range(2016,2023):
    print(year)

    ########################## UNCOMMENT WHAT DATASET TO USE
    #dataset = '../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
    #dataset = '../../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
    dataset = '../../data/conus404/wrf2d_d01_'+str(year)+'.nc'
    ##############################################################################
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
    
    ds_daily = precip.rolling(time=window).sum()

    size_sub_lat = int(len(precip.latitude)/4)
    size_sub_lon = int(len(precip.longitude)/4)

    s = storms[storms.year==year]

    # create dictionary for coordinates to reference original coordinates
    expand_lat = {}

    for i,lat in enumerate(np.sort(storms.latitude.unique())):
        expand_lat[lat] = precip.latitude.values.reshape(int(len(precip.latitude)/size_sub_lat), size_sub_lat)[i]

    expand_lon = {}

    for i,lon in enumerate(np.sort(storms.longitude.unique())):
        expand_lon[lon] = precip.longitude.values.reshape(int(len(precip.longitude)/size_sub_lon), size_sub_lon)[i]

    s = s.groupby(['latitude','longitude','id_var']).agg(list).reset_index()
    region = s.groupby(['latitude','longitude']).max().reset_index()[['latitude','longitude']]
    region['region'] = region.index
    s = pd.merge(s, region, on=['latitude', 'longitude'], how='left')

    inside_high = []

    #quantile = np.arange(0,.25+.125/3,.125/3)
    percent = np.arange(.1,.75,.2)
    for i in range(len(s)):
        sample = ds_daily.sel(time=s.iloc[i].time,
                            longitude = slice(np.min(expand_lon[s.iloc[i].longitude]),np.max(expand_lon[s.iloc[i].longitude])),latitude = slice(np.min(expand_lat[s.iloc[i].latitude]),np.max(expand_lat[s.iloc[i].latitude])))

        sample = sample.expand_dims({"storm_id": [int(s.iloc[i].id_var)]})
        sample = sample.expand_dims({"region": [int(s.iloc[i].region)]})
        sample = sample.expand_dims({"year": [int(year)]})

        add_quantile = []
        #for q in quantile:
        for q in percent:
            #threshold = annual_max[(annual_max.latitude==s.iloc[i].latitude)&(annual_max.longitude==s.iloc[i].longitude)].groupby(['latitude','longitude']).median().quantile(q,axis=1).values[0]
            threshold = (annual_max[(annual_max.latitude==s.iloc[i].latitude)&(annual_max.longitude==s.iloc[i].longitude)].groupby(['latitude','longitude']).median().mean(axis=1)*q).values[0]
            sample_above = xr.where(sample.accum >= threshold, 1, 0).sum(dim='time')

            sample_above = sample_above.expand_dims({"threshold": [threshold]})
            sample_above = sample_above.expand_dims({"quant": [q]})
            add_quantile.append(sample_above)
        
        ds = xr.merge(add_quantile).to_dataframe()
        ds = ds[ds.accum>0].reset_index()
        
        # add max intensity at each coord
        sample_max = sample.max(dim='time').to_dataframe().reset_index().drop(columns=['storm_id','year','region']).rename(columns={'accum':'max_precip'})

        ds = pd.merge(ds, sample_max, on=['latitude', 'longitude'], how='left')

        inside_high.append(ds)

    df = pd.concat(inside_high)
    #df.reset_index(drop=True).to_feather('../../output/duration_'+str(window)+'/'+name+'_'+str(year)+'_duration_above_'+str(window)+'hr')
    df.reset_index(drop=True).to_feather('../../output/duration_'+str(window)+'/'+name+'_'+str(year)+'_lower_duration_above_'+str(window)+'hr')


# %%
