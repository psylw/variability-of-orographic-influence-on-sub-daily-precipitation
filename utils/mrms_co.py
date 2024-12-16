#%%
import pandas as pd
import xarray as xr
import glob


#%%
files = glob.glob('../data/mrms/*.grib2')
# %%
for file in files:

    ##############################################################################
    precip = xr.open_dataset(file)
    precip['longitude'] = precip.longitude-360

    precip = precip.sel(longitude = slice(-109,-104),latitude = slice(41,37))

    precip.to_netcdf(file[:-6]+'_CO.nc')

#%%
def preprocess(ds):
    # Expand the 'time' dimension if it doesn't exist
    return ds.expand_dims(dim='time')
#%%
for year in range(2016,2023):
    print(year)
    files = glob.glob('../data/mrms/*'+str(year)+'*.nc')

    ds = xr.open_mfdataset(files, preprocess=preprocess)

    ds.to_netcdf('../data/mrms/'+str(year)+'_mrms_1hr_radaronly_JJA.nc')

# %%
for year in range(2016,2023):
    print(xr.open_dataset('../data/mrms/'+str(year)+'_mrms_1hr_radaronly_JJA.nc'))
# %%
