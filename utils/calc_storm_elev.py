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

            med_elev_20 = pd.merge(m_storm_20, noband, on=['latitude', 'longitude'], how='left').value.median()
            med_elev_30 = pd.merge(m_storm_30, noband, on=['latitude', 'longitude'], how='left').value.median()
            med_elev_40 = pd.merge(m_storm_40, noband, on=['latitude', 'longitude'], how='left').value.median()


            storm_attr.append([year, month, storm_id, med_elev_20,med_elev_30,med_elev_40])


        output = pd.DataFrame(data = storm_attr,columns=['year', 'month', 'storm_id', 'median_elevation20','median_elevation30','median_elevation40'])
        
        output.to_feather('..//output//'+str(year)+month+'_storm_elev')
# %%
