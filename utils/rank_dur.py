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
#%%
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
#name = 'nldas'
name = 'conus'
#%%
df_nldas = xr.open_dataset('../data/NLDAS/NLDAS_FORA0125_H.A2016.nc')
df_nldas = df_nldas.rename({'lat': 'latitude', 'lon': 'longitude'})
df_aorc = xr.open_dataset('../data/aorc/larger_aorc_APCP_surface_2016.nc')
df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016.nc')

regridder_aorc = xe.Regridder(df_aorc, df_nldas, "bilinear")
regridder_conus = xe.Regridder(df_conus, df_nldas, "bilinear")
#%%
all_years = []
for idx,year in enumerate(range(2016,2023)):
    print(year)
    ########################## UNCOMMENT WHAT DATASET TO USE
    #dataset = '../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
    #dataset = '../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
    dataset = '../data/conus404/wrf2d_d01_'+str(year)+'.nc'
    ##############################################################################
    precip = xr.open_dataset(dataset)
    ########################## UNCOMMENT WHAT DATASET TO USE
    #precip = regridder_aorc(precip)
    precip = regridder_conus(precip)
    #######################################################################
    ########################## IF NLDAS UNCOMMENT
    #precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
    ##############################################################################
    ########################## UNCOMMENT WHAT DATASET TO USE
    #precip = precip.rename({'APCP_surface': 'accum'})
    #precip = precip.rename({'Rainf': 'accum'})
    precip = precip.rename({'ACRAINLSM': 'accum'})
    #######################################################################
    precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

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
ds.to_netcdf('../output/'+name+'_rank2016.nc')
# %%
