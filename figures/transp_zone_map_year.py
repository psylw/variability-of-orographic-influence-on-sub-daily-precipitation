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
import matplotlib.gridspec as gridspec
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature
import cartopy.crs as ccrs

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
# Open the GeoTIFF file as an xarray dataset
ds = xr.open_rasterio('..//transposition_zones_shp//trans_zones.tif')
ds['x'] = ds.x+360
ds = ds.sel(y=month.latitude,x=month.longitude,method='nearest',drop=True)

ds = ds.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))

ds = ds.to_dataframe(name='value').reset_index()


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
columns = ['area_above40','time_above40', 'area40_tot']

name = [ 'area above 40 mm/hr','time above 40 mm/hr', 'fraction of area above 40 mm/hr']
#%%
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
cmap_data = [
(255,255,255),
#(255,255,217),
(237,248,177),
(199,233,180),
(127,205,187),
(65,182,196),
(29,145,192),
(34,94,168),
(37,52,148),
(8,29,88),]

cmap_data_N=[]
for i in cmap_data:
    cmap_data_N.append([c/255 for c in i])

cmap2 = LinearSegmentedColormap.from_list('custom',cmap_data_N)

for index,column in enumerate(columns):
       col = [column,'max_intersecting_geometry_id']
       n = name[index]
       data = df[df[column]>0]
       data = df[df['year']>2020]
       med = data[column].median()
       data = data.groupby(['zone_name','year']).median()[col].reset_index()
       data[column] = data[column]/med
       data[column].hist()

       if index==0:
              levels =  list(np.arange(0.1,3.1,.5))
       elif index==1:
              levels =  list(np.arange(0.1,2.1,.3))
       else:
              levels =  list(np.arange(0.05,2.1,.3))
       '''       if index<=3:
              levels =  list(np.arange(0.25,3.25,.25))
       elif index<=7:
              levels =  list(np.arange(0.8,1.5,.1))
       else:
              levels =  list(np.arange(0.1,2.1,.3))'''

       plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)

       #fig,axs = plt.subplots(3,3, subplot_kw=dict(projection=plotcrs), figsize=(16*.65,15*.8))
       fig,axs = plt.subplots(1,3, subplot_kw=dict(projection=plotcrs), figsize=(10,4))

       fig.suptitle(n,size=16)

       axs = axs.ravel()
       for idx,year in enumerate(range(2021,2024)):
              sample = data.loc[(data.year==year)]

              burn_transp = ds.merge(sample[col], left_on='value', right_on='max_intersecting_geometry_id', how='left')

              plot = burn_transp.groupby(['latitude','longitude']).max()[column].to_xarray()
              y,x = plot.latitude,plot.longitude

              axs[idx].set_extent((-109.2, -103.8, 36.9, 41.1))
              axs[idx].set_title(str(year))
              elev=axs[idx].contourf(x,y,plot.values, origin='upper', transform=ccrs.PlateCarree(),extend='both',cmap =cmap2,levels=levels)
              elev2=axs[idx].contour(noband.longitude,noband.latitude,noband.values, origin='upper', transform=ccrs.PlateCarree(),colors='grey',levels=list(np.arange(2500,4000,500)),linewidths=.25)

              #cb =plt.colorbar(elev, orientation="horizontal",pad=0.01,ax=axs[idx], shrink=.70)
              #cb.ax.tick_params(labelsize=10)

              axs[idx].add_feature(cfeature.STATES, linewidth=1)

              gdf.plot(ax = axs[idx],edgecolor='red', linewidth=.5,facecolor='none',transform=ccrs.PlateCarree())
       fig.tight_layout()

       fig.subplots_adjust(right=0.9)
       cbar_ax = fig.add_axes([.9, 0.15, 0.03, 0.7])
       #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
       fig.colorbar(elev, cax=cbar_ax)
       
       
       plt.show()



# %%
