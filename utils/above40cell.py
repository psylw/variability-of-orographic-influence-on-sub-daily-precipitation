
#%%
# calculate area, mean time, mean elev, transposition zone, mean latitude, mean longitude

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
storm_folder = os.path.join('..', '..','..','data',"storm_stats")
file_storm = glob.glob(storm_folder+'//'+'*coord40')

# open elevation
# open mrms
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

noband = noband.to_dataframe(name='value').reset_index()
#%%
all_months=[]
for file in file_storm:
    storm_attr=[]
    precip = pd.read_feather(file)
    print(file)
    for storm_id in precip.storm_id:
        index = precip.loc[precip.storm_id==storm_id]
        d = {'time':index.time.values[0],'latitude':index.latitude.values[0],'longitude':index.longitude.values[0]}
        m_storm = pd.DataFrame(data=d)
        mean_lat = m_storm.latitude.mean()
        mean_lon = m_storm.longitude.mean()

        m_storm_lat_lon_tot = m_storm.groupby(['latitude','longitude']).count()

        area_above = len(m_storm_lat_lon_tot)
        time_above = (m_storm_lat_lon_tot.time*2).mean()

        mean_elev = pd.merge(m_storm, noband, on=['latitude', 'longitude'], how='left').value.mean()

        point_series = gpd.GeoSeries([Point(x, y) for x, y in zip(m_storm.longitude-360,m_storm.latitude)])

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

        points_inside = max(intersection_counts.values())/len(point_series)

        storm_attr.append([m_storm.time[0], storm_id, max_intersecting_geometry_id, mean_lat,mean_lon, mean_elev, area_above,time_above,points_inside])


    output = pd.DataFrame(data = storm_attr,columns=['start', 'storm_id', 'max_intersecting_geometry_id', 'mean_lat','mean_lon', 'mean_elev', 'area_above','time_above','points_inside'])
    all_months.append(output)
    #output.to_feather('..//output//'+str(year)+month+'_storm_attributes')
# %%
