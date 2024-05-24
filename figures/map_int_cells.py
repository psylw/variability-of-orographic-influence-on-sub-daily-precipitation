#%%
import rioxarray as rxr
import matplotlib.gridspec as gridspec
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
import glob
import xarray as xr

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

#%%
all_months = pd.read_feather('../utils/above30_60') 
all_months['year'] = [all_months.start[i].year for i in all_months.index]
#all_months2 = pd.read_feather('../figures/above30_30_add') 
#all_months = pd.concat([all_months,all_months2],axis=1)
#%%
data=all_months[all_months.mean_rqi>.8]
#above_10 = data.loc[(data.area_above>10)&(data.area_above<1000)]
data = data[data.max_area>10]


min_size = 0.1
max_size = 500
test = data.footprint/data.mean_time_above
z_normalized = (test - test.min()) / (test.max() - test.min())  # Normalize z values to [0, 1]
sizes = min_size + z_normalized * (max_size - min_size)
data['area_norm'] = sizes
'''min_size = 0.1
max_size = 100
z_normalized = (data.max_time - data.max_time.min()) / (data.max_time.max() - data.max_time.min())  # Normalize z values to [0, 1]
sizes = min_size + z_normalized * (max_size - min_size)
data['time_norm'] = sizes'''


df = []
for year in range(2021,2024):
    yr = data[data.year == year]
    df.append(yr.groupby(pd.Grouper(key='start', freq='Y')).agg(list))

df = pd.concat(df)
#%%
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
fig,axs = plt.subplots(1,1, subplot_kw=dict(projection=plotcrs), figsize=(12,8))

elev=plt.contourf(noband.longitude,noband.latitude,noband.values, levels=list(range(2000, 5000, 500)), origin='upper',cmap='terrain', 
            alpha=0.2,transform=ccrs.PlateCarree(),extend='both')
'''elev2=axs.contour(noband.longitude,noband.latitude,noband.values, origin='upper', transform=ccrs.PlateCarree(),colors='grey',levels=list(np.arange(2500,4000,500)),linewidths=.25)'''

"""colors = ['red','green','blue']

for i in range(3):
    plt.scatter(df.iloc[i].max_area_lon, df.iloc[i].max_area_lat,transform=ccrs.PlateCarree(), s = df.iloc[i]['area_norm'],facecolor='none',edgecolors=colors[i],linewidths=.5)"""

plt.scatter(data.max_area_lon, data.max_area_lat,transform=ccrs.PlateCarree(), s = data['area_norm'],facecolor='none',edgecolors='blue',linewidths=.1)
#plt.scatter(data.max_area_lon, data.max_area_lat,transform=ccrs.PlateCarree(),c='blue',s=.1,marker='+',linewidths=.1)
'''plt.scatter(data.end_lon, data.end_lat,c='red',s=.1,marker='+',edgecolors=None,transform=ccrs.PlateCarree())

plt.scatter(data.start_lon, data.start_lat,c='green',s=.1,marker='+',edgecolors=None,transform=ccrs.PlateCarree())

data = data.reset_index(drop=True)
for i in range(len(data)):
    plt.plot([data.start_lon[i],data.max_area_lon[i],data.end_lon[i]],[data.start_lat[i],data.max_area_lat[i],data.end_lat[i]], color='cornflowerblue', linestyle='-', linewidth=.07,transform=ccrs.PlateCarree())'''



fig.tight_layout()
fig.subplots_adjust(right=.85)

axs.add_feature(cfeature.STATES, linewidth=1)


axs.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')


gl = axs.gridlines(crs=ccrs.PlateCarree(), 
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

gl = axs.gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0, 'fontsize':12}
gl.ylabel_style = {'fontsize':12}
fig.tight_layout()
fig.subplots_adjust(right=.85)
cbar_ax = fig.add_axes([.8, 0.15, 0.03, 0.7])
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
cb=fig.colorbar(elev, cax=cbar_ax)

cb.ax.tick_params(labelsize=12)
cb.set_label("elevation (m)", fontsize=12)

fig.savefig('../fig_output/map_events_se60.png',
              bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
