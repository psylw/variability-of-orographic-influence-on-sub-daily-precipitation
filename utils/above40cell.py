
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
file_storm = glob.glob(storm_folder+'//'+'*coord40')[-15:]

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

rqi_folder = '..\\..\\data\\MRMS\\RQI_2min_cat_month_CO\\'
file_rqi = glob.glob(rqi_folder+'*.grib2')[-15:]
#%%
all_months=[]
for file in range(len(file_storm)):
    storm_attr=[]
    precip = pd.read_feather(file_storm[file])
    rqi = xr.open_dataset(file_rqi[file])
    print(file)
    for storm_id in precip.storm_id:
        index = precip.loc[precip.storm_id==storm_id]
        d = {'time':index.time.values[0],'latitude':index.latitude.values[0],'longitude':index.longitude.values[0]}
        
        m_storm = pd.DataFrame(data=d)
        mean_lat = m_storm.latitude.mean()
        mean_lon = m_storm.longitude.mean()
        
        m_storm_lat_lon_tot = m_storm.groupby(['latitude','longitude']).count()

        footprint = len(m_storm_lat_lon_tot)
        mean_time_above = (m_storm_lat_lon_tot.time*2).mean()

        time_group = pd.concat([m_storm.groupby('time').count(),m_storm.groupby('time').agg(list).rename(columns={'latitude':'lat_list','longitude':'lon_list'})],axis=1)

        # area at start
        start_area = time_group.iloc[0].latitude
         # lat/lon start area
        start_lat = np.mean(time_group.iloc[0].lat_list)
        start_lon = np.mean(time_group.iloc[0].lon_list)

        # area at end
        end_area = time_group.iloc[-1].latitude
         # lat/lon end area
        end_lat = np.mean(time_group.iloc[-1].lat_list)
        end_lon = np.mean(time_group.iloc[-1].lon_list)

        # max area
        max_area = time_group.latitude.max()
        # lat/lon max area
        max_area_lat = np.mean(time_group[time_group.latitude==max_area].lat_list.iloc[0])
        max_area_lon = np.mean(time_group[time_group.latitude==max_area].lon_list.iloc[0])

        # max time
        max_time = (m_storm_lat_lon_tot.time*2).max()
        # lat/lon max time
        max_time_lat = m_storm_lat_lon_tot[m_storm_lat_lon_tot.time==m_storm_lat_lon_tot.time.max()].index[0][0]
        max_time_lon = m_storm_lat_lon_tot[m_storm_lat_lon_tot.time==m_storm_lat_lon_tot.time.max()].index[0][1]

        mean_elev = pd.merge(m_storm, noband, on=['latitude', 'longitude'], how='left').value.mean()

        m_storm['fill']=1
        test = m_storm.groupby(['time','latitude','longitude']).max().to_xarray()

        try:
            mean_rqi = rqi.where(test.fill==1).unknown.mean().values
        except ValueError as e:
            print("Variable is NaN. Handling NaN case...")
            mean_rqi = np.nan

        storm_attr.append([m_storm.time[0], storm_id, mean_lat,mean_lon, footprint,mean_time_above,start_area,start_lat,start_lon, end_area,end_lat,end_lon, max_area,max_area_lat,max_area_lon,max_time,max_time_lat,max_time_lon,mean_elev,mean_rqi])        

    output = pd.DataFrame(data = storm_attr,columns=['start', 'storm_id', 'mean_lat','mean_lon', 'footprint','mean_time_above','start_area','start_lat','start_lon', 'end_area','end_lat','end_lon', 'max_area','max_area_lat','max_area_lon','max_time','max_time_lat','max_time_lon','mean_elev','mean_rqi'])

    all_months.append(output)

# %%
all_months = pd.concat(all_months).reset_index()
all_months['mean_rqi']=all_months['mean_rqi'].astype('float')
all_months.to_feather('above30_60')
# %%
