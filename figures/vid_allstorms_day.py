
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

# %%

all_months = pd.read_feather('../utils/above30_30') 
data=all_months[all_months.mean_rqi>.8]
above_10 = data[data.area_above>10]

min_size = 0.1
max_size = 100

z_normalized = (above_10.area_above - above_10.area_above.min()) / (above_10.area_above.max() - above_10.area_above.min())  # Normalize z values to [0, 1]
sizes = min_size + z_normalized * (max_size - min_size)
above_10['size'] = sizes

df = []
for year in range(2021,2024):
    yr = above_10[above_10.year == year]
    df.append(yr.groupby(pd.Grouper(key='start', freq='d')).agg(list))

df = pd.concat(df)
#%%
# with elevation
fig, ax = plt.subplots()
sc = ax.scatter([], [], zorder=2)
# Plot the elevation data as the background
lon, lat = np.meshgrid(noband.x, noband.y)
elevation_plot = ax.pcolormesh(lon,lat,noband, shading='auto', cmap='terrain',alpha=0.4)
# Function to initialize the plot
def init():
    ax.set_xlim(251, 256)
    ax.set_ylim(37, 41)

    return sc,

# Function to update the plot for each frame
def update(frame):
    print(f"{df.index[frame]}"[0:10])
    t = df.iloc[frame]
    
    if not t.empty:
        offsets = np.column_stack((t['mean_lon'], t['mean_lat']))
        sc.set_offsets(offsets)
        sc.set_sizes(np.ravel(t['size']))
        sc.set_edgecolor('red')
        sc.set_facecolor('none')
        sc.set_linewidths(0.5)
    else:
        sc.set_offsets(np.zeros((0, 2)))   # Clear scatter if there's no data for the current frame
        sc.set_sizes(np.array([]))
    
    ax.set_title(f"{df.index[frame]}"[0:10])
    return sc,

# Create animation
ani = FuncAnimation(fig, update, frames=len(df), init_func=init,blit=True)

ani.save('lat_lon_animation_day.mp4', fps=1)

plt.show()