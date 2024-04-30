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

# Open the GeoTIFF file as an xarray dataset
ds = xr.open_rasterio('..//transposition_zones_shp//trans_zones.tif')
ds = ds.to_dataframe(name='value').value.reset_index()

files = glob.glob('..\\output\\*tes')

all = []
for file in files:
       df = pd.read_feather(file)
       all.append(df)
#%%
df = pd.concat(all).reset_index()       
shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)

df['zone_name'] = [gdf.loc[gdf.TRANS_ZONE==df.max_intersecting_geometry_id[i]].ZONE_NAME.values[0] for i in range(len(df))]

columns = ['area_above20', 'area_above30', 'area_above40', 
       'time_above20', 'time_above30', 'time_above40']

name = [ 'area above 20 mm/hr', 'area above 30 mm/hr', 'area above 40 mm/hr', 
       'time above 20 mm/hr', 'time above 30 mm/hr', 'time above 40 mm/hr']
#%%
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
cmap_data = [
(255,255,217),
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

col1 = 'area_above20'
col = [col1,'max_intersecting_geometry_id']
name = 'area above 40 mm/hr'
data = df.groupby(['zone_name','year','month']).mean()[col].reset_index()
levels =  list(np.arange(0,500,50))

for month in ['may','jun','jul','aug','sep']:

       plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)

       fig,axs = plt.subplots(3,3, subplot_kw=dict(projection=plotcrs), figsize=(16*.65,15*.65))

       gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)
       fig.suptitle(month+', '+name)

       axs = axs.ravel()
       for idx,year in enumerate(range(2015,2024)):
              sample = data.loc[(data.year==year)&(data.month==month)]

              burn_transp = ds.merge(sample[col], left_on='value', right_on='max_intersecting_geometry_id', how='left').dropna()

              plot = burn_transp.groupby(['y','x']).max()[col1].to_xarray()
              y,x = plot.y,plot.x

              axs[idx].set_extent((-109.2, -103.5, 36.8, 41.3))
              axs[idx].set_title(str(year))
              elev=axs[idx].contourf(x,y,plot.values, origin='upper', transform=ccrs.PlateCarree(),extend='both',cmap =cmap2,levels=levels)

              cb =plt.colorbar(elev, orientation="horizontal",pad=0.01,ax=axs[idx])
              cb.ax.tick_params(labelsize=10)
              cb.set_label(name)

              axs[idx].add_feature(cfeature.STATES, linewidth=1)

              axs[idx].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

       plt.show()



       

# %%
