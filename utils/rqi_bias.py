#%%
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
from datetime import timedelta
# open data
rqi = xr.open_dataset('..\\..\\data\\MRMS\\RQI_cat_yr_CO\\2021_RQI_CO.grib2', chunks={'time': '500MB'})

threshold = .8

rqi = rqi.where(rqi.longitude<=256,drop=True)
rqi = rqi.where(rqi.unknown>=0)


#%%
test = rqi.where(rqi.unknown<threshold)

test = test.to_dataframe().dropna()

test = test.reset_index()
test['hour'] = test['time'].dt.hour
std_hr = test.groupby(['latitude','longitude']).std().hour
std_hr.to_xarray().plot()
#%%
test = rqi.where(rqi.unknown>threshold)

test = test.to_dataframe().dropna()

# Calculate difference in time between consecutive points within each coordinate group
df['time_diff'] = df.groupby(['lat', 'lon'])['time'].diff().dt.total_seconds() / 3600.0