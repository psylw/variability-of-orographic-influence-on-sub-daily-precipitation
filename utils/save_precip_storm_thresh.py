#%%
###############################################################################
# label each storm with area, direction, velocity, decay rate for each threshold level
###############################################################################

# open max 15-min intensity

# for each storm above threshold
# get coordinates
# get 15-min values
# calculate 'volume' or total px for storm
# select unique coordinates above threshold
# calculate 'volume' or total px above threshold
# calculate time above threshold for each px
# spatial concentration

import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# get all coordinate files

# open data
rate_folder = '..\\..\\data\\MRMS\\2min_rate_cat_month_CO\\'
storm_folder = '..\\..\\data\\storm_stats\\'
file_2min = glob.glob(rate_folder+'*.grib2')
file_storm = glob.glob(storm_folder+'//'+'*_coord*')

px_thresh = pd.read_feather('../output/count_duration_threshold_px')
# get month and year
px_thresh['year'] = [px_thresh.start_storm[i].year for i in px_thresh.index]
px_thresh['month'] = [px_thresh.start_storm[i].month for i in px_thresh.index]
# groupby month year and agg all storms
storms = px_thresh.groupby(['year','month','storm_id']).max().reset_index()
#%%

for year in range(2015,2024):
    print(year)
    for month_idx,month in enumerate(['may','jun','jul','aug','sep']):

        storms_mo_yr = storms.loc[(storms.year == year)&(storms.month == month_idx+5)]

        name_month = [s for s in file_2min if month in s and str(year) in s][0]
        print(name_month)
        precip = xr.open_dataset(name_month, chunks={'time': '500MB'}).unknown 
        precip = precip.where(precip>=0) # get rid of negatives
        precip = precip*(2/60)# get mrms into 2min accum from rate

        ########### CALC 15MIN INT
        precip = precip.resample(time='1T').asfreq().fillna(0)
        precip = precip.rolling(time=15,min_periods=1).sum()*(60/15)

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
        i=0
        storm_precip = []
        whole_storm = []
        for storm_idx in storms_mo_yr.storm_id:
            print(i/len(storms_mo_yr))
            index = s.loc[s.storm_id==storm_idx]

            d = {'time':index.time.values[0],'latitude':index.latitude.values[0],'longitude':index.longitude.values[0]}
            storm = pd.DataFrame(data=d)

            storm_total = storm.groupby(['latitude','longitude']).count()
            area_tot = len(storm_total)
            time_tot = (storm_total.time*2).mean()
            whole_storm.append([year, month, storm_idx, area_tot,time_tot])

            storm['fill'] = 1

            temp = storm.groupby(['time','latitude','longitude']).max().to_xarray()

            m_storm = precip.sel(time=temp.time,latitude=temp.latitude,longitude=temp.longitude)

            m_storm = m_storm.where(temp.fill==1)

            chunk_sizes = {dim: 'auto' for dim in m_storm.dims}
            m_storm = m_storm.chunk(chunk_sizes)
            m_storm = m_storm.where(m_storm>=20)
            m_storm = m_storm.to_dataframe().dropna().reset_index()
            m_storm['storm_id'] = storm_idx

            storm_precip.append(m_storm.drop(columns=['step','heightAboveSea']).groupby('storm_id').agg(list))

            i+=1

        output1 = pd.concat(storm_precip).reset_index()
        output1.to_feather('..//output//'+str(year)+month+'_storm_thr_precip')

        output2 = pd.DataFrame(data = whole_storm,columns=['year', 'month','storm_idx', 'area_tot','time_tot'])
        output2.to_feather('..//output//'+str(year)+month+'_storm_total')
# %%
