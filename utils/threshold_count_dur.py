#%%
###############################################################################
# count/duration above various accumulation thresholds per pixel
###############################################################################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

# %%
# open data
rate_folder = '..\\..\\data\\MRMS\\2min_rate_cat_month_CO\\'
storm_folder = '..\\..\\data\\storm_stats\\'
file_2min = glob.glob(rate_folder+'*.grib2')
file_storm = glob.glob(storm_folder+'*.nc')

thresholds = [20,30,40]

count_dur_point = []

#%%
for monthly_2min, monthly_storm in zip(file_2min, file_storm):
    precip = xr.open_dataset(monthly_2min, chunks={'time': '500MB'}).unknown 
    precip = precip.where(precip>=0) # get rid of negatives
    precip = precip*(2/60)# get mrms into 2min accum from rate
    
    ########### CALC 15MIN INT
    precip = precip.resample(time='1T').asfreq().fillna(0)
    precip = precip.rolling(time=15,min_periods=1).sum()*(60/15)

    storm = xr.open_dataset(monthly_storm, chunks={'time': '500MB'})

    ds = storm.assign(precip=precip)

    ########### SELECT EVERY 10KM (ADDED COORD ON END)
    #lat_sel = np.append(np.arange(0,len(ds.latitude),10), len(ds.latitude)-1)
    #lon_sel = np.append(np.arange(0,len(ds.longitude),10), len(ds.longitude)-1)

    #ds = ds.isel(latitude=lat_sel, longitude=lon_sel)
    ds = ds.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    ds = ds.drop(['step','heightAboveSea'])

    ds = ds.to_dataframe()
    ds = ds.reset_index()

    for threshold in thresholds:
        th = ds.loc[ds.precip >= threshold]

        for lat, lon in set(zip(th.latitude,th.longitude)): 
            point = th.loc[(th.latitude == lat) & (th.longitude== lon)]

            for storm in point.storm_id.unique():

                dur_above = len(point.loc[point.storm_id==storm])*2 # 2 minute timesteps
                start_storm = point.loc[point.storm_id==storm].time.iloc[0]
                max_int = point.loc[point.storm_id==storm].precip.max()

                count_dur_point.append([lat,lon,threshold,max_int,storm,dur_above,start_storm])

#%%
# save

df=pd.DataFrame(count_dur_point, columns=['latitude','longitude','threshold','max_int','storm_id','dur_above_min','start_storm'])

df = df.loc[df.storm_id!=0]

df.to_feather('../output/count_duration_threshold_px')