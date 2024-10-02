#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xesmf as xe
from matplotlib import gridspec

name = 'conus'

#%%
all_years = []
for idx,year in enumerate(range(2002,2023)):
    print(year)
    dataset = '../data/conus404/wrf2d_d01_'+str(year)+'.nc'
    precip = xr.open_dataset(dataset)

    precip = precip.rename({'ACRAINLSM': 'accum'})
    
    precip = precip.sel(longitude = slice(-109.04,-103.96),latitude = slice(36.98,41.02))
    #######################################################################
    precip = precip.where(precip>=0)

    size_sub_lat = int(len(precip.latitude)/4)
    size_sub_lon = int(len(precip.longitude)/4)

    ds_window = []
    for window in (1,3,12,24):
        ds_daily = precip.rolling(time=window).sum()
        ds_aggregated = ds_daily.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()
        ds_daily = ds_daily.max(dim='time')
        ds_aggregated = ds_aggregated.max(dim='time').to_dataframe().reset_index()

        expand_lat = {}
        for i,lat in enumerate(np.sort(ds_aggregated.latitude.unique())):
            expand_lat[lat] = ds_daily.latitude.values.reshape(int(len(ds_daily.latitude)/size_sub_lat), size_sub_lat)[i]

        expand_lon = {}
        for i,lon in enumerate(np.sort(ds_aggregated.longitude.unique())):
            expand_lon[lon] = ds_daily.longitude.values.reshape(int(len(ds_daily.longitude)/size_sub_lon), size_sub_lon)[i]

        ds_region = []
        for i in ds_aggregated.index:
            sample = ds_daily.sel(
                                longitude = slice(np.min(expand_lon[ds_aggregated.iloc[i].longitude]),np.max(expand_lon[ds_aggregated.iloc[i].longitude])),latitude = slice(np.min(expand_lat[ds_aggregated.iloc[i].latitude]),np.max(expand_lat[ds_aggregated.iloc[i].latitude])))
            
            stacked_data = sample['accum'].stack(spatial=('latitude', 'longitude'))

            # 2. Rank the values across the 'spatial' dimension
            ranked_data = stacked_data.rank(dim='spatial')

            # 3. Unstack the 'spatial' dimension to go back to the original latitude and longitude
            ranked_data_unstacked = ranked_data.unstack('spatial')

            # 4. Add the ranked data back to the original dataset
            sample['precip_ranked'] = ranked_data_unstacked
            ds_region.append(sample)
        ds_window.append(xr.merge(ds_region))
    combined = xr.concat(ds_window, dim='xarray_index')
    combined = combined.to_dataframe()
    combined['year'] = year

    all_years.append(combined)

df = pd.concat(all_years)
ds = df.reset_index().groupby(['latitude','longitude','year','xarray_index']).max().to_xarray()
ds.to_netcdf('../output/conus_rankall.nc')
# %%
