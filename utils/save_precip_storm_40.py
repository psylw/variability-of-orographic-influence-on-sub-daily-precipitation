#%%
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point,LineString

# open storms above 20 mm/hr
output_folder = '..\\output\\'
file_thr_storm = glob.glob(output_folder+'*thr_precip')[-15:]

# open data
rate_folder = '..\\..\\data\\MRMS\\2min_rate_cat_month_CO\\'
storm_folder = '..\\..\\data\\storm_stats\\'
file_2min = glob.glob(rate_folder+'*.grib2')[-15:]
file_storm = glob.glob(storm_folder+'//'+'*id.nc*')[-15:]

#%%
grid_size = 4
smallest_int = 100
storm_xr = []
storm_40 = []
for file in range(0,15):
    print(file)
    storms_above20 = pd.read_feather(file_thr_storm[file])
    storms_above20 = storms_above20.explode(['time','latitude','longitude','unknown'])
    storms_above40 = storms_above20[storms_above20.unknown>=40]
    # remove storms where area < 4
    test = storms_above40.groupby(['storm_id','latitude','longitude']).count().reset_index()

    area_above40 = []
    for storm in test.storm_id.unique():
        s = test[test.storm_id==storm]
        if len(s) > smallest_int:
            area_above40.append(storm)
    storms_above40 = storms_above40[storms_above40.storm_id.isin(area_above40)]

    test = storms_above40.groupby(['storm_id','time']).agg(list).reset_index()

    area = []
    centroid = []
    for i in test.index:
        lat_lon_list = [(lat, lon) for lat, lon in zip(test.latitude[i], test.longitude[i])]
        # If you have only two coordinates, use LineString instead of Polygon
        if len(lat_lon_list) == 1:
            area.append(1)
            centroid.append(lat_lon_list)
        elif len(lat_lon_list) <= 2:
            linestring = LineString(lat_lon_list)
            point = linestring.centroid.coords[0]
            centroid.append(point)
            length = linestring.length  # Calculate length instead of area
            area.append(2)
        else:
            # Create a shapely Polygon object from the coordinates
            polygon = Polygon(lat_lon_list)
            point = polygon.centroid

            # Get the centroid of the polygon
            centroid.append(tuple([point.x,point.y]))

            # Get the area of the polygon
            area.append(polygon.area)

    test['area'] = area
    test['centroid'] = centroid
    storm_40.append(test)

    storms_above40_id = storms_above40.storm_id.unique()

    precip = xr.open_dataset(file_2min[file], chunks={'time': '500MB'}).unknown 
    precip = precip.where(precip>=0) # get rid of negatives
    precip = precip*(2/60)# get mrms into 2min accum from rate

    ########### CALC 15MIN INT
    precip = precip.resample(time='1T').asfreq().fillna(0)
    precip = precip.rolling(time=15,min_periods=1).sum()*(60/15)

    storms = xr.open_dataset(file_storm[file],chunks={'time': '500MB'})
    storms = storms.assign(intensity=precip)
    storms = storms.isel(latitude=slice(None, None, grid_size), longitude=slice(None, None, grid_size))

    storms = storms.where(storms.storm_id.isin(storms_above40_id)).to_dataframe().reset_index()

    storms = storms.dropna().reset_index(drop=True)
    storms = storms.drop(columns=['step','heightAboveSea'])
    storms = storms.groupby(['time','latitude','longitude']).max()
    storms = storms.to_xarray()
    print(storms)
    storm_xr.append(storms)

# %%
