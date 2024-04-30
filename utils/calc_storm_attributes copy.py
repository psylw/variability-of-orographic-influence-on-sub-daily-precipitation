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

shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)

# open data
output_folder = '..\\output\\'
file_thr_storm = glob.glob(output_folder+'*thr_precip')
#%%

for year in range(2015,2024):
    print(year)
    for month_idx,month in enumerate(['may','jun','jul','aug','sep']):
        name_month = [s for s in file_thr_storm if month in s and str(year) in s][0]

        precip = pd.read_feather(name_month)

        storm_attr=[]
        for storm_id in precip.storm_id:
            index = precip.loc[precip.storm_id==storm_id]
            d = {'time':index.time.values[0],'latitude':index.latitude.values[0],'longitude':index.longitude.values[0],'intensity':index.unknown.values[0]}
            m_storm = pd.DataFrame(data=d)

            m_storm_20 = m_storm.loc[m_storm.intensity>=20]
            m_storm_30 = m_storm.loc[m_storm.intensity>=30]
            m_storm_40 = m_storm.loc[m_storm.intensity>=40]

            m_storm_lat_lon_tot_20 = m_storm_20.groupby(['latitude','longitude']).count()
            m_storm_lat_lon_tot_30 = m_storm_30.groupby(['latitude','longitude']).count()
            m_storm_lat_lon_tot_40 = m_storm_40.groupby(['latitude','longitude']).count()

            area_above20 = len(m_storm_lat_lon_tot_20)
            area_above30 = len(m_storm_lat_lon_tot_30)
            area_above40 = len(m_storm_lat_lon_tot_40)

            time_above20 = (m_storm_lat_lon_tot_20.time*2).mean()
            time_above30 = (m_storm_lat_lon_tot_30.time*2).mean()
            time_above40 = (m_storm_lat_lon_tot_40.time*2).mean()

            point_series = gpd.GeoSeries([Point(x, y) for x, y in zip(m_storm_20.longitude-360,m_storm_20.latitude)])

            # Initialize a dictionary to store the counts of points intersecting with each geometry
            intersection_counts = {}

            # Iterate over each geometry in the GeoDataFrame
            for idx, geometry in gdf.geometry.iteritems():
                # Find intersection between the geometry and the points
                intersection = point_series.intersects(geometry)
                
                # Count the number of points that intersect with the geometry
                count = intersection.sum()
                
                # Store the count in the dictionary with the geometry's unique identifier
                intersection_counts[gdf.at[idx, 'TRANS_ZONE']] = count

            # Find the geometry with the maximum count of intersecting points
            max_intersecting_geometry_id = max(intersection_counts, key=intersection_counts.get)


            storm_attr.append([year, month, storm_id, max_intersecting_geometry_id,

                            area_above20,
                            area_above30,
                            area_above40,

                            time_above20,
                            time_above30,
                            time_above40])


        output = pd.DataFrame(data = storm_attr,columns=['year', 'month', 'storm_id', 'max_intersecting_geometry_id',

                            'area_above20',
                            'area_above30',
                            'area_above40',

                            'time_above20',
                            'time_above30',
                            'time_above40'])
        
        output.to_feather('..//output//'+str(year)+month+'_storm_attributes')
# %%
