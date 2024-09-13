#%%
import earthaccess
#pip install earthaccess
import xarray as xr
import os
import requests
from bs4 import BeautifulSoup
import glob
# Authenticate with NASA Earthdata using earthaccess
auth = earthaccess.login(strategy="interactive")  # Change to 'interactive' if no .netrc file
precip=xr.open_dataset('../data/aorc/larger_aorc_APCP_surface_'+str(2023)+'.nc')
lat_min = int(precip.latitude.min().values)
lat_max = int(precip.latitude.max().values)+.5
lon_left = int(precip.longitude.min().values)-.5
lon_right = int(precip.longitude.max().values)+.5

#%%
from datetime import datetime, timedelta

# Define start and end years
start_year = 1979
end_year = 2023

# List to store (year, day_of_year) tuples for June, July, August
year_day_list = []

# Loop through the years
for year in range(start_year, end_year + 1):
    # Start from June 1st and go through August 31st
    start_date = datetime(year, 6, 1)
    end_date = datetime(year, 9, 1)
    
    # Loop through each day from June 1 to August 31
    delta = timedelta(days=1)
    while start_date < end_date:
        day_of_year = start_date.timetuple().tm_yday
        # Format day of the year with leading zeros (e.g., 001, 002, ...)
        day_of_year_str = f"{day_of_year:03d}"
        year_day_list.append((year, day_of_year_str))
        start_date += delta

#%%
#for year, _ in year_day_list:
#    print(year)

year = 1979
for _, day in year_day_list:
    # Specify the URL
    url = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/'+str(year)+'/'+str(day)+'/'

    # Send a request to the URL
    response = requests.get(url)

    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the links that end with '.nc'
    nc_files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]

    for file in nc_files:
        # Define the output directory and filename
        output_dir = '../data/NLDAS/'
        url_file = url+file
        os.makedirs(output_dir, exist_ok=True)
        temp_file_path = os.path.join(output_dir, url_file.split('/')[-1])

        # Download the file using earthaccess
        earthaccess.download(url_file, local_path=output_dir)

        # Open the downloaded file using xarray
        ds = xr.open_dataset(temp_file_path)

        ds_selected = ds.Rainf.isel(time=0).sel(lat = slice(lat_min,lat_max), lon = slice(lon_left,lon_right))

        filename = os.path.splitext(temp_file_path)[0] + '_CO.nc'
        ds_selected.to_netcdf(filename)

        ds.close()
        ds_selected.close()

        os.remove(temp_file_path)
#%%
# Define the directory where the files are located and the pattern
file_pattern = '../data/NLDAS/*_CO.nc'

# Use glob to find all matching files
file_list = glob.glob(file_pattern)

# Load multiple NetCDF files into an xarray dataset
ds = xr.open_mfdataset(file_list, combine='by_coords')
filename_year = 
ds.to_netcdf(filename_year)
ds.close()

for file_name in file_list:
    os.remove(file_name)
# %%
