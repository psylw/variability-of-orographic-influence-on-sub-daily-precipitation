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

#%%

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

q1 = noband.quantile(.333)
q2 = noband.quantile(.667)

map1 = noband.where(noband<q1)
map2 = noband.where((noband>q1) &(noband<q2))
map3 = noband.where(noband>q2)
#%%

gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
fig,axs = plt.subplots(1,3, subplot_kw=dict(projection=plotcrs), figsize=(15*.6,7.8*.6))

lon, lat = np.meshgrid(noband.x,noband.y)
levels = list(range(1500, 5000, 500))
elev=axs[0].contourf(lon,lat, map1, levels=levels, origin='upper',cmap='terrain', 
            alpha=0.8,transform=ccrs.PlateCarree(),extend='both')
elev=axs[1].contourf(lon,lat, map2, levels=levels, origin='upper',cmap='terrain', 
            alpha=0.8,transform=ccrs.PlateCarree(),extend='both')
elev=axs[2].contourf(lon,lat, map3, levels=levels, origin='upper',cmap='terrain', 
            alpha=0.8,transform=ccrs.PlateCarree(),extend='both')
fig.tight_layout()
fig.subplots_adjust(right=.85)
cbar_ax = fig.add_axes([.9, 0.15, 0.03, 0.7])
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
cb=fig.colorbar(elev, cax=cbar_ax)

cb.ax.tick_params(labelsize=12)
cb.set_label("elevation (m)", fontsize=12)

axs[0].add_feature(cfeature.STATES, linewidth=1)
axs[1].add_feature(cfeature.STATES, linewidth=1)
axs[2].add_feature(cfeature.STATES, linewidth=1)

axs[0].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')
axs[1].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')
axs[2].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

gl = axs[0].gridlines(crs=ccrs.PlateCarree(), 
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

gl = axs[1].gridlines(crs=ccrs.PlateCarree(), 
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
gl.left_labels=False # suppress right labels

gl = axs[2].gridlines(crs=ccrs.PlateCarree(), 
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
gl.left_labels=False # suppress right labels
#plt.legend()

fig.savefig("../fig_output/map_bands.png",
           bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
