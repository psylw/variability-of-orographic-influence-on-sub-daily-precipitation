#%%
import xarray as xr
import os
import glob
#%%
# Directory containing your NetCDF files
#input_dir = "../data/conus404"
#output_dir = "../data/conus404/seasonal"

input_dir = "../data/conus404/PREC_ACC_NC"
output_dir = "../data/conus404/PREC_ACC_NC_season"
os.makedirs(output_dir, exist_ok=True)

# Define years and seasonal months
#years = range(2016, 2022)
years = range(2019, 2023)
seasons = {
    "DJF": [("12", -1), ("01", 0), ("02", 0)],  # December (previous year), January, February
    "MAM": [("03", 0), ("04", 0), ("05", 0)],  # March, April, May
    "JJA": [("06", 0), ("07", 0), ("08", 0)],  # June, July, August
    "SON": [("09", 0), ("10", 0), ("11", 0)],  # September, October, November
}

# Loop through each year and season
for year in years:
    for season, months in seasons.items():
        # Initialize a list to collect datasets for the season
        seasonal_datasets = []

        for month, year_offset in months:
            # Adjust the year for December (previous year)
            file_year = year + year_offset
            # Find files matching the current year and month
            file_pattern = os.path.join(input_dir, f"wrf2d_d01_{file_year}-{month}-*.nc")
            files = glob.glob(file_pattern)
            
            if files:
                # Load all files for the current year and month into a single xarray.Dataset
                ds = xr.open_mfdataset(files, combine='by_coords')
                seasonal_datasets.append(ds)
            else:
                print(f"No files found for {file_year}-{month}")

        # Concatenate all monthly datasets for the season if any files were found
        if seasonal_datasets:
            ds_season = xr.concat(seasonal_datasets, dim="time")
            
            # Define the output filename
            output_file = os.path.join(output_dir, f"wrf2d_d01_{year}_{season}.nc")
            
            # Save the filtered dataset
            ds_season.to_netcdf(output_file)
            print(f"Saved {season} data for {year} to {output_file}")
            
            # Close the dataset
            ds_season.close()

# %%
