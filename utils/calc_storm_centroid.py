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
from shapely.geometry import MultiPoint

# open data
output_folder = '..\\output\\'
file_thr_storm = glob.glob(output_folder+'*thr_precip')
#%%

for year in range(2021,2024):
    print(year)
    for month_idx,month in enumerate(['may','jun','jul','aug','sep']):
        name_month = [s for s in file_thr_storm if month in s and str(year) in s][0]

        precip = pd.read_feather(name_month)

        storm_attr=[]
        for storm_id in precip.storm_id:
            index = precip.loc[precip.storm_id==storm_id]
            d = {'time':index.time.values[0],'latitude':index.latitude.values[0],'longitude':index.longitude.values[0],'intensity':index.unknown.values[0]}
            m_storm = pd.DataFrame(data=d)

            m_storm_40 = m_storm.loc[m_storm.intensity>=40]
            x = m_storm_40.longitude.values
            y = m_storm_40.latitude.values
            try:
                points = gpd.GeoSeries.from_xy(x, y)

                centroid = MultiPoint(points).centroid
                centroid_lat = centroid.y
                centroid_lon = centroid.x

                storm_attr.append([year, month, storm_id, centroid_lat,centroid_lon])
            except:
                pass

        output = pd.DataFrame(data = storm_attr,columns=['year', 'month', 'storm_id', 'centroid40_lat', 'centroid40_lon'])
        
        output.to_feather('..//output//'+str(year)+month+'_storm_centroid')
# %%
