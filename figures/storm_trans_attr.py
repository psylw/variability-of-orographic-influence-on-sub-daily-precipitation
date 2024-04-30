#%%
import glob
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import xarray as xr

files1 = glob.glob('..\\output\\*tes')
files2 = glob.glob('..\\output\\*tal')

all = []
for file1,file2 in zip(files1,files2):
       p1 = pd.read_feather(file1)
       p2 = pd.read_feather(file2)

       p2 = p2.rename(columns={'storm_idx':'storm_id'})

       all.append(pd.merge(p1, p2, on=['year', 'month', 'storm_id'], how='left'))
#%%
df = pd.concat(all).reset_index().fillna(0)        
shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)

df['zone_name'] = [gdf.loc[gdf.TRANS_ZONE==df.max_intersecting_geometry_id[i]].ZONE_NAME.values[0] for i in range(len(df))]

df['area20_tot'] = df.area_above20/df.area_tot
df['area30_tot'] = df.area_above30/df.area_tot
df['area40_tot'] = df.area_above40/df.area_tot

columns = ['area_above20', 'area_above30', 'area_above40', 'area_tot',
       'time_above20', 'time_above30', 'time_above40', 'time_tot','area20_tot','area30_tot','area40_tot']

name = [ 'area above 20 mm/hr', 'area above 30 mm/hr', 'area above 40 mm/hr','total area', 
       'time above 20 mm/hr', 'time above 30 mm/hr', 'time above 40 mm/hr', 'total time','area20_tot','area30_tot','fraction of area above 40 mm/hr']
#%%
# open transposition zone as tif
ds = xr.open_rasterio('..//transposition_zones_shp//trans_zones.tif')
ds = ds.to_dataset(name='value')

datafile1 = "../../../data/elev_data/CO_SRTM1arcsec__merge.tif"
codtm =xr.open_rasterio(datafile1)
newelev = codtm.drop_vars('band')
noband = newelev.sel(band=0)

# open mrms
data_folder = os.path.join('..', '..','..','data','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')
month = xr.open_dataset(filenames[0], engine = "cfgrib",chunks={'time': '500MB'})

month = month.where(month.longitude<=256,drop=True)

ds['x'] = ds.x+360
noband['x'] = noband.x+360
ds = ds.sel(y=month.latitude,x=month.longitude,method='nearest',drop=True)
noband = noband.sel(y=month.latitude,x=month.longitude,method='nearest',drop=True)

ds['elevation'] = noband
# %%
elev_zone_mean = []
col = 'area_above40'
data = df[df[col]>0]

for zone in df.max_intersecting_geometry_id.unique():
    elev = ds.where(ds.value==zone).elevation.max().values-ds.where(ds.value==zone).elevation.min().values

    plot = data[data.max_intersecting_geometry_id==zone]
    elev_zone_mean.append([plot[col].median(),elev,zone])

elev_zone_mean = pd.DataFrame(elev_zone_mean,columns=['median_area','max_elev','zone'])
plt.scatter(elev_zone_mean.max_elev,elev_zone_mean.median_area)

