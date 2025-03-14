#%%
import os
from datetime import datetime
import xarray as xr
#%%
def add_time_dimension(ds):
    filepath = ds.encoding['source']
    filename = os.path.basename(filepath)
    date_str = filename.split('_')[4]  # Extract 'YYYYMMDD' part
    date = datetime.strptime(date_str, '%Y%m%d')
    return ds.assign_coords(time=date).expand_dims('time')

co_lat_min = 36.9
co_lon_min = 250.8-360
co_lat_max = 41.1
co_lon_max = 256-360
import glob
import pandas as pd

input_dir = "../data/prism"
output_dir = "../data/prism/seasonal"
os.makedirs(output_dir, exist_ok=True)

seasons = {
    "DJF": [("12", -1), ("01", 0), ("02", 0)],  # December (previous year), January, February
    "MAM": [("03", 0), ("04", 0), ("05", 0)],  # March, April, May
    "JJA": [("06", 0), ("07", 0), ("08", 0)],  # June, July, August
    "SON": [("09", 0), ("10", 0), ("11", 0)],  # September, October, November
}
#%%
for year in range(1981, 2023):
    for season, months in seasons.items():
        datasets = []
        for month, year_offset in months:
            file_year = year + year_offset
            file_pattern = os.path.join(input_dir, f"PRISM_ppt_stable_4kmD2_{file_year}{month}*.bil")
            files = glob.glob(file_pattern)
            for file in files:
                ds = xr.open_dataset(file, engine='rasterio')
                ds = add_time_dimension(ds)
                datasets.append(ds)
        if datasets:
            combined = xr.concat(datasets, dim='time')
            combined = combined.rename({'y': 'latitude','x':'longitude'}).isel(band=0)
            combined = combined.sortby("latitude", ascending=False).sel(latitude=slice(co_lat_max,co_lat_min),longitude=slice(co_lon_min,co_lon_max))
            
            output_file = os.path.join(output_dir, f"PRISM_ppt_{year}_{season}.nc")
            combined.to_netcdf(output_file)
            print(f"Saved {season} data for {year} to {output_file}")

# %%
