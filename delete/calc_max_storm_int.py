#%%
###############################################################################
# calculate the maximum intensity for each storm
###############################################################################

import xarray as xr
import numpy as np
import os
import glob
import pandas as pd

# get all coordinate files

# get all rate files
mrms_folder = os.path.join('..', '..','..','data',"MRMS","2min_rate_cat_month_CO")
storm_folder = os.path.join('..', '..','..','data',"storm_stats")

file_mrms = glob.glob(mrms_folder+'//'+'*.grib2')
file_storm = glob.glob(storm_folder+'//'+'*_coord*')

px_thresh = pd.read_feather('../output/count_duration_threshold_px')
# get month and year
px_thresh['year'] = [px_thresh.start_storm[i].year for i in px_thresh.index]
px_thresh['month'] = [px_thresh.start_storm[i].month for i in px_thresh.index]
# groupby month year and agg all storms
storms = px_thresh.groupby(['year','month']).agg(list).storm_id.reset_index()
#%%
max_int = []
for year in range(2015,2024):
    print(year)
    for month_idx,month in enumerate(['may','jun','jul','aug','sep']):
        name_month = [s for s in file_mrms if month in s and str(year) in s][0]
        print(name_month)
        m = xr.open_dataset(name_month, chunks={'time': '50MB'})

        name_month = [s for s in file_storm if month in s and str(year) in s]
        if len(name_month)==1:
            s = pd.read_feather(name_month[0])
        else:
            s1 = pd.read_feather(name_month[0])
            s2 = pd.read_feather(name_month[1])
            s = pd.concat([s1,s2])
            s = s.groupby('storm_id').agg({'time': lambda x: [item for sublist in x for item in sublist],
                                  'latitude': lambda x: [item for sublist in x for item in sublist],
                                  'longitude': lambda x: [item for sublist in x for item in sublist]}).reset_index()

        # select storms from px threshold df
        storms_thresh = storms.loc[(storms.year==year)&(storms.month==month_idx+5)].storm_id.iloc[0]
        storms_thresh = np.unique(storms_thresh)
        # get mrms into 2min accum from rate
        m = m.where(m.longitude<=256,drop=True)
        m = m.where(m>=0)
        m = m*(2/60)

        ########### CALC 16MIN INT
        m = m.resample(time='1T').asfreq().fillna(0) # don't resample to 1min, rough 15min calc
        m = m.rolling(time=15,min_periods=1).sum()*(60/15)
        m = m.where(m>=20)

        max_15min = []

        i=0
        for storm in storms_thresh:
            print(i/len(s))
            index = s.iloc[storm]
            d = {'time':index.time,'latitude':index.latitude,'longitude':index.longitude}
            temp = pd.DataFrame(data=d)

            temp = temp.groupby(['time','latitude','longitude']).max().to_xarray()

            m_storm = m.sel(time=temp.time,latitude=temp.latitude,longitude=temp.longitude)

            # sum across all coordinates, get temporal variance
            max_15min.append([year, month, index.storm_id,float(m_storm.unknown.max().values)])

            i+=1

        max_15min = pd.DataFrame(max_15min,columns=['year','month','storm_id','max_15min']).fillna(0)
        max_15min = max_15min.loc[max_15min.max_15min>=20]
        max_int.append(max_15min)
#%%
# save
# label max threshold
        
max_int = pd.concat(max_int)
max_int.to_feather('../storm_output/max_15min_above10')


# %%
