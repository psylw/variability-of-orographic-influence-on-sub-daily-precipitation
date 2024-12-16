# open REFS, ATLAS-14, and CONUS404
# save 10-yr with elevation
# %%
import re
import xarray as xr
import numpy as np
import glob
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
import xesmf as xe

#%%

file_paths = glob.glob("../data/atlas14/*.asc")

all_years= []
yr = 10

# Load each file and add to a list of DataArrays with duration as a dimension
data_arrays = []
for asc_file in file_paths:
    # Extract year and duration from the filename
    match = re.search(r"mw(\d+)yr(\d+)", asc_file)
    if match:
        year = int(match.group(1))
        duration = int(match.group(2))
        if "ma" in asc_file:
            duration = duration / 60

        if year != yr:
            continue  # Skip files where year is not 1
    else:
        raise ValueError(f"Filename {asc_file} does not match expected format.")
    
    # Load your actual data from .asc file here; here, we'll use a placeholder 2D array
    with rasterio.open(asc_file) as src:
        # Read the data
        data = src.read(1)  # Read the first (and usually only) band
        
        # Get the x and y coordinates based on the affine transformation
        # Generate coordinate arrays using the width and height of the data
        transform = src.transform
        x_coords = np.array([transform * (col, 0) for col in range(src.width)])[:, 0]
        y_coords = np.array([transform * (0, row) for row in range(src.height)])[:, 1]
        
        # Create the xarray DataArray
        da = xr.DataArray(data, coords={"y": y_coords, "x": x_coords, "duration": duration}, dims=("y", "x"), attrs=src.meta)

        clipped_ds = da.sel(y=slice(41,37),x=slice(-109,-104.005))  # Replace with actual data loading

    data_arrays.append(clipped_ds)

ds = xr.concat(data_arrays, dim="duration")
ds =  ds.where(ds.duration.isin([1,24]),drop=True)
ds = ds.rename({"x": "longitude", "y": "latitude"})
# from metadata, Grid cell precipitation in inches*1000
ds = ds*(25.4/1000)
#%%
df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016_JJA.nc')
df_conus = df_conus.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))

regridder = xe.Regridder(ds, df_conus, "conservative")
atlas = regridder(ds)
atlas = atlas.to_dataframe(name='accum').reset_index()
atlas['dataset'] = 'atlas14'

atlas_1 = atlas[atlas.duration==1].rename(columns={'accum':'accum_1hr'}).drop(columns='duration')

atlas_24 = atlas[atlas.duration==24].rename(columns={'accum':'accum_24hr'}).drop(columns='duration')

pd.merge(atlas_1,atlas_24,on=['latitude','longitude','dataset']).to_feather('../output/atlas_14')

#%%
file_paths = glob.glob("../data/refs/*.tif")

def get_duration(filename):
    # Find the duration in the filename, e.g., '0015m' or '0002h'
    duration_match = re.search(r'(\d+)(m|h)', filename)
    if duration_match:
        duration_value = int(duration_match.group(1))
        duration_unit = duration_match.group(2)
        # Convert minutes to hours if necessary
        if duration_unit == 'm':
            return duration_value / 60  # Convert minutes to hours
        return duration_value  # Duration is already in hours
    return None  # Return None if no duration found

# Process each file individually
datasets = []
for file in file_paths:
    if '10-1' in file:
        # Open the file as an xarray dataset
        ds = xr.open_dataset(file)
        ds = ds.sel(band=1).band_data
        ds = ds.rio.reproject("EPSG:4326")
        ds = ds.sel(y=slice(41,37),x=slice(-109,-104.005))
        ds = ds.rename({"x": "longitude", "y": "latitude"})

        regridder = xe.Regridder(ds, df_conus, "conservative")
        ds = regridder(ds)
        ds['duration'] = get_duration(file)
        
        datasets.append(ds)

refs = xr.concat(datasets, dim="duration")
refs =  refs.where(refs.duration.isin([1,24]),drop=True)
# convert to mm
refs =  refs*(25.4)
refs = refs.to_dataframe(name='accum').reset_index()
refs['dataset'] = 'refs'


refs_1 = refs[refs.duration==1].rename(columns={'accum':'accum_1hr'}).drop(columns='duration')

refs_24 = refs[refs.duration==24].rename(columns={'accum':'accum_24hr'}).drop(columns='duration')

pd.merge(refs_1,refs_24,on=['latitude','longitude','dataset']).to_feather('../output/reps')