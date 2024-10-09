#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import xesmf as xe
from matplotlib import gridspec


#%%
all_years = []
dataset = '../output/mrms_ann_max_px_window_60.nc'

precip = xr.open_dataset(dataset)

for idx,year in enumerate(range(2016,2023)):
    print(year)
    precip_year = precip.sel(year=idx)
    
    size_sub_lat = int(len(precip_year.latitude)/4)
    size_sub_lon = int(len(precip_year.longitude)/4)

    
    ds_aggregated = precip_year.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max().to_dataframe().reset_index()

    expand_lat = {}
    for i,lat in enumerate(np.sort(ds_aggregated.latitude.unique())):
        expand_lat[lat] = precip_year.latitude.values.reshape(int(len(precip_year.latitude)/size_sub_lat), size_sub_lat)[i]

    expand_lon = {}
    for i,lon in enumerate(np.sort(ds_aggregated.longitude.unique())):
        expand_lon[lon] = precip_year.longitude.values.reshape(int(len(precip_year.longitude)/size_sub_lon), size_sub_lon)[i]

        ds_region = []
        for i in ds_aggregated.index:
            sample = precip_year.sel(
                                longitude = slice(np.min(expand_lon[ds_aggregated.iloc[i].longitude]),np.max(expand_lon[ds_aggregated.iloc[i].longitude])),latitude = slice(np.min(expand_lat[ds_aggregated.iloc[i].latitude]),np.max(expand_lat[ds_aggregated.iloc[i].latitude])))
            
            stacked_data = sample['accum'].stack(spatial=('latitude', 'longitude'))

            # 2. Rank the values across the 'spatial' dimension
            ranked_data = stacked_data.rank(dim='spatial')

            # 3. Unstack the 'spatial' dimension to go back to the original latitude and longitude
            ranked_data_unstacked = ranked_data.unstack('spatial')

            # 4. Add the ranked data back to the original dataset
            sample['precip_ranked'] = ranked_data_unstacked
            sample = sample.to_dataframe().reset_index()
            
            ds_region.append(sample)
        ds_region = pd.concat(ds_region)
        #ds_region['window'] = window
        #ds_window.append(ds_region)

    #combined = pd.concat(ds_window)
    #combined['year'] = year

    all_years.append(ds_region)

#df = pd.concat(all_years)

# %%
