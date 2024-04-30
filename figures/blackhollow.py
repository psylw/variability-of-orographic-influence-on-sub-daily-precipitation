#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from shapely.geometry import MultiPoint
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature

shapefile_path = "../black_hollow_shp/globalwatershed.shp"
gdf = gpd.read_file(shapefile_path)

bh = pd.read_feather('..\\output\\2021jul_storm_thr_precip')

black_hollow = bh[bh.storm_id==165098.0]

d = {'time':black_hollow.time.values[0],'latitude':black_hollow.latitude.values[0],'longitude':black_hollow.longitude.values[0],'unknown':black_hollow.unknown.values[0]}
storm = pd.DataFrame(data=d)
#storm = storm[storm.unknown>40]

temp = storm.groupby(['latitude','longitude']).max().to_xarray()
#%%

fig = plt.figure(1, figsize=(8,8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
ax = plt.subplot(1,1,1, projection=plotcrs)

gdf.plot(ax = ax,edgecolor='red', linewidth=1.5,facecolor='none',transform=ccrs.PlateCarree())


elev=ax.contourf(temp.longitude,temp.latitude,temp.unknown, origin='upper',cmap='terrain', 
            alpha=0.4,transform=ccrs.PlateCarree(),extend='both')

cb =fig.colorbar(elev,orientation="horizontal", shrink=.55,pad=0.01)
cb.ax.tick_params(labelsize=12)
cb.set_label("max 15-minute intensity, mm/hr", fontsize=12)
ax.set_extent((-106.5,-105.5, 40.4, 40.9))
gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0, 'fontsize':12}
gl.ylabel_style = {'fontsize':12}
# add these before plotting
gl.bottom_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels
# %%
