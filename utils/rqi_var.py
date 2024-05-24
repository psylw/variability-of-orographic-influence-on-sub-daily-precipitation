#%%
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd

# open data
rqi_folder = '..\\..\\data\\MRMS\\RQI_2min_cat_month_CO\\'
file_rqi = glob.glob(rqi_folder+'*.grib2')[-15:]

threshold = .8
#%%
# first figure out area, based on min month timesteps > threshold
i=0
for file in file_rqi:
    rqi = xr.open_dataset(file, chunks={'time': '500MB'})
    rqi = rqi.where(rqi.longitude<=256,drop=True)
    rqi = rqi.where(rqi.unknown>=0)
    
    test = rqi.where(rqi.unknown>threshold)
    test = test.count(dim='time') 
    test = test.to_dataframe().reset_index()
    test.to_feather('month_count_above_rqi'+str(i))
    i+=1
#%%
# combine months together, select coordinates based on month with min values above threshold
file_area = glob.glob('month'+'*')
area = []
for file in file_area:
    area.append(pd.read_feather(file))
area = pd.concat(area)
area = area.groupby(['latitude','longitude']).min() # minimum monthly timesteps above 0.8
#%%
area = area.to_xarray()

#%%
# then remove times

# remove timesteps lower than threshold
# Threshold value
threshold = .7
masked = []
for file in file_rqi:
    
    rqi = xr.open_dataset(file, chunks={'time': '500MB'})
    rqi_area = rqi.where(area.unknown>5000)
    # Check if any value is below the threshold along the 'time' dimension
    above_threshold = ~(rqi_area['unknown'] < threshold).any(dim=['latitude','longitude'])

    ds_masked = rqi_area.where(above_threshold, drop=True)
    masked.append(ds_masked)
    print(len(ds_masked.time))

# combine and save
masked = xr.concat(masked,dim='time')
# %%
