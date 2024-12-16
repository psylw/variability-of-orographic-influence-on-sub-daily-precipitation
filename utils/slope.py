#%%
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import rasterio

import xesmf as xe
#%%
df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016_JJA.nc')
df_conus = df_conus.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))
#%%

# Load the source raster (the one to be resampled)
slope = rioxarray.open_rasterio("../data/slope.tif").isel(band=0).rename({"y": "latitude", "x": "longitude"})

target_raster = df_conus.rio.write_crs("EPSG:4326")

resampled_raster = slope.rio.reproject_match(target_raster, resampling=rasterio.enums.Resampling.bilinear)
# %%
resampled_raster.rename({"y": "latitude", "x": "longitude"}).to_dataframe(name='slope').drop(columns=['band','spatial_ref']).to_feather('../output/conus_slope')


# %%
aspect = rioxarray.open_rasterio("../data/aspect.tif").isel(band=0).rename({"y": "latitude", "x": "longitude"})

aspect.to_dataframe(name='aspect').drop(columns=['band','spatial_ref']).to_feather('../output/conus_aspect')
# %%
