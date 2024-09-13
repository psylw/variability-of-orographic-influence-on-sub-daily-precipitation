# get the duration above annual max intensity 
#%%
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import rasterio


#%%
# open elevation and resample to precip grid
precip=xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_'+str(2023)+'.nc')
# Load the source raster (the one to be resampled)

source_raster = rioxarray.open_rasterio("../../../../data/elev_data/CO_SRTM1arcsec__merge.tif")

# Ensure the target raster has georeferencing information
target_raster = precip.isel(time=0).sel(longitude = slice(-109,-104),latitude = slice(37,41)).rio.write_crs("EPSG:4326")

# Resample the source raster to the target raster's grid using bilinear interpolation
resampled_raster = source_raster.rio.reproject_match(target_raster, resampling=rasterio.enums.Resampling.bilinear)

resampled_raster=resampled_raster.isel(band=0).where(resampled_raster>0).to_dataframe(name='elevation').drop(columns=['spatial_ref']).reset_index().rename(columns={'y':'latitude','x':'longitude'})
# %%
for window in (1,3,12,24):
    storms = pd.read_feather('../../output/storms_above_thr'+'_window_'+str(window))
    storms = storms.drop(columns='index')
    storms['year'] = storms.time.dt.year

    annual_max = pd.read_feather('../../output/ann_max_region'+'_window_'+str(window))
    for year in range(1979,2024):
        print(year)

        precip = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc')

        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

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

        quantile = np.arange(0,1,.125)
        
        for i in range(len(s)):
            sample = ds_daily.sel(time=s.iloc[i].time,
                                longitude = slice(np.min(expand_lon[s.iloc[i].longitude]),np.max(expand_lon[s.iloc[i].longitude])),latitude = slice(np.min(expand_lat[s.iloc[i].latitude]),np.max(expand_lat[s.iloc[i].latitude])))
            sample = sample.expand_dims({"storm_id": [int(s.iloc[i].id_var)]})
            sample = sample.expand_dims({"region": [int(s.iloc[i].region)]})
            sample = sample.expand_dims({"year": [int(year)]})

            add_quantile = []
            for q in quantile:

                threshold = annual_max[(annual_max.latitude==s.iloc[i].latitude)&(annual_max.longitude==s.iloc[i].longitude)].groupby(['latitude','longitude']).median().quantile(q,axis=1).values[0]

                sample_above = xr.where(sample.APCP_surface >= threshold, 1, 0).sum(dim='time')

                sample_above = sample_above.expand_dims({"threshold": [threshold]})
                sample_above = sample_above.expand_dims({"quant": [q]})
                add_quantile.append(sample_above)
            
            ds = xr.merge(add_quantile).to_dataframe()
            ds = ds[ds.APCP_surface>0].reset_index()
            
            # add max intensity at each coord
            sample_max = sample.max(dim='time').to_dataframe().reset_index().drop(columns=['storm_id','year','region']).rename(columns={'APCP_surface':'max_precip'})
            ds = pd.merge(ds, sample_max, on=['latitude', 'longitude'], how='left')
            # add elevation at each coord
            ds = pd.merge(ds, resampled_raster.drop(columns='band'), on=['latitude', 'longitude'], how='left')

            inside_high.append(ds)

        df = pd.concat(inside_high)

        df.reset_index(drop=True).to_feather('../../output/duration_'+str(window)+'/'+str(year)+'_duration_above_'+str(window)+'hr')

# %%
