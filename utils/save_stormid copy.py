# %%
###############################################################################
# assign storm-id using connected component analysis
###############################################################################

# create unique storm label for spatially connected regions where precip>0. 
# use class/patch_stormid_acrossmonths.py to patch storm ids across months
import xarray as xr
import numpy as np
import pandas as pd
import skimage.measure
from skimage.morphology import remove_small_objects
import pyarrow.feather as feather
from dask.distributed import Client
import shutil 
import os
import glob

# %%
def find_storms(mrms_ds):
    # create binary ndarray
    a = xr.where(mrms_ds >= 60, 1, 0)
    # label objects
    storm_id, count_storms = skimage.measure.label(a, connectivity=1, return_num=True)
    # remove object SMALLER than 5 px
    #storm_id = remove_small_objects(storm_id, 9)
    
    return storm_id

# Create a path to the code file
data_folder = os.path.join('..', '..','..','data','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')[-15:]

destination = os.path.join('..', '..','..','data','storm_stats')

#client = Client()
'''file_area = glob.glob('month'+'*')
area = []
for file in file_area:
    area.append(pd.read_feather(file))
area = pd.concat(area)
area = area.groupby(['latitude','longitude']).min()

area = area.to_xarray()'''


#%%
hdf_fail = []

for idx,file in enumerate(filenames):
    print(idx)
    month = xr.open_dataset(file, engine = "cfgrib",chunks={'time': '500MB'})

    month = month.where(month.longitude<=256,drop=True)

    month = month.where(month>=0) # get rid of negatives
    month = month*(2/60)# get mrms into 2min accum from rate

    ########### CALC 15MIN INT
    #month = month.resample(time='1T').asfreq().fillna(0)
    month = month.rolling(time=30,min_periods=1).sum()*(60/30)

    #month = month.where(area.unknown>5000) # approx 0.25 timesteps

    month = month.unknown

    storm_id = find_storms(month)

    storm_id = storm_id.astype('float32')
    # create dataset from storm_id
    time = month.time
    latitude = month.latitude 
    longitude = month.longitude

    ds = xr.Dataset(data_vars=dict(storm_id=(["time", "latitude", "longitude"], storm_id),),
        coords=dict(time=time,latitude=latitude,longitude=longitude,))

    name = '//'+file[-22:-6]+'_storm_id_30min_60'
    path = destination+name+'.nc'


    ds.to_netcdf(path=path)

# %%
storm_folder = os.path.join('..', '..','..','data',"storm_stats")

file_storm = glob.glob(storm_folder+'//'+'*30min_60.nc')[-15:]
#%%
for file in file_storm:
    storm = xr.open_dataset(file)
    storm = storm.where(storm>0,drop=True)
    name = '//'+file[-37:-12]+'_coord30min_60'
    storm = storm.to_dataframe()
    storm = storm.loc[storm.storm_id>0]
    storm = storm.reset_index().groupby(['storm_id']).agg(list)
    storm = storm.reset_index()

    storm.to_feather(storm_folder+name)
# %%
# open data
# SAVE PRECIP IN HIGH INT CELL
rate_folder = '..\\..\\data\\MRMS\\2min_rate_cat_month_CO\\'
storm_folder = '..\\..\\data\\storm_stats\\'
file_2min = glob.glob(rate_folder+'*.grib2')[-15:]
file_storm = glob.glob(storm_folder+'//'+'*40.nc*')[-15:]


all_cells = pd.read_feather('../utils/above30_30') 
above_10 = all_cells[all_cells.area_above>=10]
above_10 = above_10[above_10.mean_rqi>=.8]

#%%
storm_xr=[]
for file in range(0,15):
    
    precip = xr.open_dataset(file_2min[file], chunks={'time': '500MB'}).unknown 
    precip = precip.where(precip>=0) # get rid of negatives
    precip = precip*(2/60)# get mrms into 2min accum from rate

    precip = precip.rolling(time=30,min_periods=1).sum()*(60/30)

    storms = xr.open_dataset(file_storm[file],chunks={'time': '500MB'})
    storms = storms.assign(intensity=precip)

    # only select storms where footprint > 10 km2
    select_storms = above_10.loc[above_10.start.isin(precip.time.values)].storm_id

    test = storms.where(storms.storm_id.isin(select_storms),drop=True)
    storm_xr.append(test)
#%%
#%%