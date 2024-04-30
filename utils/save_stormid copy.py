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
    a = xr.where(mrms_ds >= 40, 1, 0)
    # label objects
    storm_id, count_storms = skimage.measure.label(a, connectivity=1, return_num=True)
    # remove object SMALLER than 5 px
    #storm_id = remove_small_objects(storm_id, 9)
    
    return storm_id


# Create a path to the code file
data_folder = os.path.join('..', '..','..','data','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')

destination = os.path.join('..', '..','..','data','storm_stats')

#client = Client()


#%%
hdf_fail = []

for idx,file in enumerate(filenames):
    print(idx)
    #month = xr.open_dataset(file, engine = "cfgrib",chunks={'time': '500MB'})
    month = xr.open_dataset(file, engine = "cfgrib",chunks={'time': '500MB'})
    #print(month)
    month = month.where(month.longitude<=256,drop=True)
    #month = month.unknown.where(month.unknown > 0, drop=True)
    month = month.unknown

    storm_id = find_storms(month)

    storm_id = storm_id.astype('float32')
    # create dataset from storm_id
    time = month.time
    latitude = month.latitude 
    longitude = month.longitude

    ds = xr.Dataset(data_vars=dict(storm_id=(["time", "latitude", "longitude"], storm_id),),
        coords=dict(time=time,latitude=latitude,longitude=longitude,))

    name = '//'+file[-22:-6]+'_storm_id40'
    path = destination+name+'.nc'

    #ds.to_netcdf(path=path)
    #ds.close()
    try:
        ds.to_netcdf(path=path)
    except Exception as e:
        seg = np.array_split(ds.time.values,3)
        for i in range(3):
            name = name+str(i)
            path = destination+name+'.nc'
            ds.sel(time = time.isin(seg[i])).to_netcdf(path=path)







# %%
