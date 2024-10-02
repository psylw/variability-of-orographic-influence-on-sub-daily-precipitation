#%%
import xarray as xr
#import requests
import time
import glob
from urllib.parse import urlparse
import os
import numpy as np
import xesmf as xe
#%%
precip=xr.open_dataset('../data/aorc/larger_aorc_APCP_surface_'+str(2023)+'.nc')
lat_min = int(precip.latitude.min().values)
lat_max = int(precip.latitude.max().values)+.5
lon_left = int(precip.longitude.min().values)-.5
lon_right = int(precip.longitude.max().values)+.5

# Define the grid spacing for approximately 4 km resolution
lat_spacing = 0.036  # 4 km spacing in latitude (degrees)
lon_spacing = 0.047  # 4 km spacing in longitude (degrees)

ds_out = xr.Dataset(
    {
        "latitude": (["latitude"], np.arange(lat_min, lat_max, lat_spacing)),
        "longitude": (["longitude"], np.arange(lon_left, lon_right, lon_spacing)),
    }
)

# make regridder to apply to all 
url = 'https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000/wy1995/199508/wrf2d_d01_1995-08-01_01:00:00.nc?Time[0:1:0],XLAT[0:1:1014][0:1:1366],XLONG[0:1:1014][0:1:1366],ACRAINLSM[0:1:0][0:1:1014][0:1:1366]'

ds = xr.open_dataset(url)

regridder = xe.Regridder(ds, ds_out, "bilinear")

output_dir = '../data/conus404/'

max_retries = 5
retry_delay = 10
#%%
base_url = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000/"
months_30days = ["06"]
months_31days = ["07", "08"]
years = range(1983,2002)  
hours = [f"{hour:02d}" for hour in range(24)]  # Generates hours 00 to 23
template_url = "wy{year}/{year}{month}/wrf2d_d01_{year}-{month}-{day}_{hour}:00:00.nc?Time[0:1:0],XLAT[0:1:1014][0:1:1366],XLONG[0:1:1014][0:1:1366],ACRAINLSM[0:1:0][0:1:1014][0:1:1366]"

for year in years:
    print(year)
    file_list = []
    # Handle months with 30 days (June, September)
    for month in months_30days:
        for day in range(1, 31):  # Days 1 to 30
            for hour in hours:
                file_url = base_url + template_url.format(year=year, month=month, day=f"{day:02d}", hour=hour)
                file_list.append(file_url)
    
    # Handle months with 31 days (July, August)
    for month in months_31days:
        for day in range(1, 32):  # Days 1 to 31
            for hour in hours:
                file_url = base_url + template_url.format(year=year, month=month, day=f"{day:02d}", hour=hour)
                file_list.append(file_url)

    # Output the list or save it to a file

    for url in file_list:
        # Open the dataset using xarray
        for attempt in range(max_retries):
            try:
                # Try to open the dataset
                ds = xr.open_dataset(url)

                ds = regridder(ds)

                ds = ds.rename({'Time': 'time'})

                ds_selected = ds.ACRAINLSM
                
                parsed_url = urlparse(url)
                filename = parsed_url.path.split('/')[-1].split('.')[0][0:-6]+'_CO.nc'
                ds_selected.to_netcdf(output_dir+filename)
                ds.close()
                ds_selected.close()
                
                # If successful, break the loop
                #print("Dataset successfully opened!")
                break
            except OSError as e:
                # Print error and retry after waiting
                #print(f"Attempt {attempt+1} failed with error: {e}")
                
                if attempt < max_retries - 1:
                    #print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Could not open the dataset.")
                    print(url)
                    raise
        
    # Define the directory where the files are located and the pattern
    file_pattern = '../data/conus404/*_CO.nc'

    # Use glob to find all matching files
    filelist = glob.glob(file_pattern)

    # Load multiple NetCDF files into an xarray dataset
    ds = xr.open_mfdataset(filelist, combine='by_coords', chunks=None,parallel=True)

    filename_year = f'wrf2d_d01_{year}.nc'
    ds.to_netcdf(output_dir+filename_year)
    ds.close()

    for file in filelist:
        os.remove(file)



