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
precip=xr.open_dataset('../data/aorc/larger_aorc_APCP_surface_'+str(2022)+'.nc')
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
#url = 'https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000/wy1995/199508/wrf2d_d01_1995-08-01_01:00:00.nc?Time[0:1:0],XLAT[0:1:1014][0:1:1366],XLONG[0:1:1014][0:1:1366],ACRAINLSM[0:1:0][0:1:1014][0:1:1366]'
url = 'https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000/wy1995/199508/wrf2d_d01_1995-08-01_01:00:00.nc?Time[0:1:0],XLAT[0:1:1014][0:1:1366],XLONG[0:1:1014][0:1:1366],PREC_ACC_NC[0:1:0][0:1:1014][0:1:1366]'
ds = xr.open_dataset(url)

regridder = xe.Regridder(ds, ds_out, "bilinear")

#output_dir = '../data/conus404/'
output_dir = '../data/conus404/PREC_ACC_NC/'

max_retries = 5
retry_delay = 10
#%%
base_url = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000/"
months_30days = ["04", "06", "09", "11"]
months_31days = ["01", "03", "05", "07", "08", "10", "12"]
february = "02"
years = range(2021, 2023)  
hours = [f"{hour:02d}" for hour in range(24)]  # Generates hours 00 to 23
#template_url = "wy{water_year}/{calendar_year}{month}/wrf2d_d01_{calendar_year}-{month}-{day}_{hour}:00:00.nc?Time[0:1:0],XLAT[0:1:1014][0:1:1366],XLONG[0:1:1014][0:1:1366],ACRAINLSM[0:1:0][0:1:1014][0:1:1366]"
template_url = "wy{water_year}/{calendar_year}{month}/wrf2d_d01_{calendar_year}-{month}-{day}_{hour}:00:00.nc?Time[0:1:0],XLAT[0:1:1014][0:1:1366],XLONG[0:1:1014][0:1:1366],PREC_ACC_NC[0:1:0][0:1:1014][0:1:1366]"

for water_year in years:
    file_list = []
    # Months October to December of the previous calendar year
    previous_year = water_year - 1
    for month in ["10", "11", "12"]:
        days_in_month = 30 if month in months_30days else 31
        for day in range(1, days_in_month + 1):
            for hour in hours:
                file_url = base_url + template_url.format(
                    water_year=water_year,
                    calendar_year=previous_year,
                    month=month,
                    day=f"{day:02d}",
                    hour=hour
                )
                file_list.append(file_url)
    
    # Months January to September of the current water year
    for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
        # Adjust for February (leap years)
        if month == february:
            days_in_february = 29 if (water_year % 4 == 0 and (water_year % 100 != 0 or water_year % 400 == 0)) else 28
            days_in_month = days_in_february
        else:
            days_in_month = 30 if month in months_30days else 31
        for day in range(1, days_in_month + 1):
            for hour in hours:
                file_url = base_url + template_url.format(
                    water_year=water_year,
                    calendar_year=water_year,
                    month=month,
                    day=f"{day:02d}",
                    hour=hour
                )
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

                #ds_selected = ds.ACRAINLSM
                ds_selected = ds.PREC_ACC_NC

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
    """    # Define the directory where the files are located and the pattern
    file_pattern = '../data/conus404/*_CO.nc'

    # Use glob to find all matching files
    filelist = glob.glob(file_pattern)

    # Load multiple NetCDF files into an xarray dataset
    ds = xr.open_mfdataset(filelist, combine='by_coords', chunks=None,parallel=True)

    filename_year = f'wrf2d_d01_{water_year}.nc'
    ds.to_netcdf(output_dir+filename_year)
    ds.close()

    for file in filelist:
        os.remove(file)
"""




# %%
