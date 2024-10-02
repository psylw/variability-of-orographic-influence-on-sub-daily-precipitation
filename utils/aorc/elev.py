#%%
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import rasterio

#%%
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
#name = 'nldas'
name = 'conus'

########################## UNCOMMENT WHAT DATASET TO USE
#dataset = '../../data/aorc/larger_aorc_APCP_surface_'+str(2022)+'.nc'
#dataset = '../../data/NLDAS/NLDAS_FORA0125_H.A'+str(2000)+'.nc'
dataset = '../../data/conus404/wrf2d_d01_'+str(2022)+'.nc'
##############################################################################
precip = xr.open_dataset(dataset)
########################## IF NLDAS UNCOMMENT
#precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
##############################################################################

# Load the source raster (the one to be resampled)
source_raster = rioxarray.open_rasterio("../../../../data/elev_data/CO_SRTM1arcsec__merge.tif")

##############################################################################
########################## USE FOR NLDAS AND AORC
#target_raster = precip.isel(time=0).sel(longitude = slice(-109,-104),latitude = slice(37,41)).rio.write_crs("EPSG:4326")
########################## IF CONUS UNCOMMENT
target_raster = precip.isel(time=0).sel(longitude = slice(-109.04,-103.96),latitude = slice(36.98,41.02)).rio.write_crs("EPSG:4326")
##############################################################################

# Resample the source raster to the target raster's grid using bilinear interpolation
resampled_raster = source_raster.rio.reproject_match(target_raster, resampling=rasterio.enums.Resampling.bilinear)

resampled_raster=resampled_raster.isel(band=0).where(resampled_raster>0).to_dataframe(name='elevation').drop(columns=['spatial_ref']).reset_index().rename(columns={'y':'latitude','x':'longitude'})
#%%
##############################################################################
########################## USE FOR NLDAS AND AORC
#precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))
########################## IF CONUS UNCOMMENT
precip = precip.sel(longitude = slice(-109.04,-103.96),latitude = slice(36.98,41.02))
##############################################################################

size_sub_lat = int(len(precip.latitude)/4)
size_sub_lon = int(len(precip.longitude)/4)

ds_aggregated = precip.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()
ds_aggregated = ds_aggregated.fillna(0)

ds_aggregated = ds_aggregated.max(dim='time')
storms = ds_aggregated.to_dataframe().reset_index()
storms['region'] = storms.index
#%%
# create dictionary for coordinates to reference original coordinates
expand_lat = {}

for i,lat in enumerate(np.sort(storms.latitude.unique())):
    expand_lat[lat] = precip.latitude.values.reshape(int(len(precip.latitude)/size_sub_lat), size_sub_lat)[i]

expand_lon = {}

for i,lon in enumerate(np.sort(storms.longitude.unique())):
    expand_lon[lon] = precip.longitude.values.reshape(int(len(precip.longitude)/size_sub_lon), size_sub_lon)[i]
#%%
resampled_raster = resampled_raster.drop(columns='band').groupby(['latitude','longitude']).max().to_xarray()

add_region = []

for i in storms.index:
    sample = resampled_raster.sel(
                        longitude = slice(np.min(expand_lon[storms.iloc[i].longitude]),np.max(expand_lon[storms.iloc[i].longitude])),latitude = slice(np.min(expand_lat[storms.iloc[i].latitude]),np.max(expand_lat[storms.iloc[i].latitude])))
    new_variable = xr.DataArray(storms.iloc[i].region, dims=('latitude', 'longitude'),
                            coords={'latitude': sample['latitude'], 'longitude': sample['longitude']})
    sample['region'] = new_variable
    add_region.append(sample)
add_region = xr.merge(add_region)
#%%
df = add_region.to_dataframe()

df_elev = []
for region in df.region.unique():
    sample = df[(df.region==region)]
    sample['elevation_category'] = pd.qcut(sample['elevation'], q=4, labels=['Low', 'Lower Middle', 'Upper Middle', 'High'])

    df_elev.append(sample)
df = pd.concat(df_elev).reset_index()
# %%
df.to_feather('../../output/'+name+'_elev')