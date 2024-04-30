#%%
###############################################################################
# count/duration above various accumulation thresholds per pixel
###############################################################################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import timedelta

# open data
rate_folder = '..\\..\\data\\MRMS\\2min_rate_cat_month_CO\\'
file_2min = glob.glob(rate_folder+'*.grib2')

count_dur_point = []

#%%
for idx,monthly_2min in enumerate(file_2min):
    precip = xr.open_dataset(monthly_2min, chunks={'time': '500MB'}).unknown 
    precip = precip.where(precip>=0) # get rid of negatives
    precip = precip.where(precip.longitude<=256,drop=True)
    precip = precip*(2/60)# get mrms into 2min accum from rate
    
    ########### CALC 15MIN INT
    precip = precip.resample(time='1T').asfreq().fillna(0)
    precip = precip.rolling(time=15,min_periods=1).sum()*(60/15)
    
    precip = precip.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    precip = precip.where(precip>=20)

    #precip = precip.resample(time='5T').max()

    precip = precip.to_dataframe()
    precip = precip.dropna().reset_index()

    print(idx/len(file_2min))
    count_dur_point.append(precip)

#%%

df = pd.concat(count_dur_point)


# %%
# add storm id, same storm id if break < break_time
break_time = 8 # hours
storm_id = []
df = df.drop(columns=['step', 'heightAboveSea'])
df = df.sort_values(by=['time'])

grouped = df.groupby(['latitude','longitude']).agg(list).reset_index()

for i in grouped.index:
    timestamps = grouped.time.iloc[i]

    id_list = []
    current_id = 1
    last_timestamp = timestamps[0]

    for timestamp in timestamps:
        if timestamp - last_timestamp > timedelta(hours=break_time):
            current_id += 1

        id_list.append(current_id)
        last_timestamp = timestamp

    storm_id.append(id_list)

grouped = grouped.explode(['time','unknown'])
grouped['storm_id'] = np.concatenate(storm_id)
#%%
# calculate duration of storm, minutes
times_perloc_perstorm = grouped.groupby(['latitude','longitude','storm_id']).count().time.reset_index()
times_perloc_perstorm = times_perloc_perstorm.rename(columns={'time':'duration'})
total_storms = pd.merge(grouped,times_perloc_perstorm,on=['latitude','longitude','storm_id'])
#%%
# add elevation
# open mrms
import os
data_folder = os.path.join('..', '..','..','data','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')
month = xr.open_dataset(filenames[0], engine = "cfgrib",chunks={'time': '500MB'})

month = month.where(month.longitude<=256,drop=True)

datafile1 = "../../../data/elev_data/CO_SRTM1arcsec__merge.tif"
codtm =xr.open_rasterio(datafile1)
newelev = codtm.drop_vars('band')
noband = newelev.sel(band=0)
noband['x'] = noband.x+360
noband = noband.sel(y=month.latitude,x=month.longitude,method='nearest',drop=True)

elevation = noband.to_dataframe(name='value').reset_index()

grouped = pd.merge(grouped, elevation, on=['latitude', 'longitude'], how='left')
#############################################################################
#%%
#save
grouped.reset_index(drop=True).to_feather('../output/count_duration_threshold_px2')
# %%
