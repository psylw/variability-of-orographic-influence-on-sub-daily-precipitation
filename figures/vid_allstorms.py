
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
# create timeline
from datetime import datetime, timedelta
start_date = datetime(2023, 5, 1)
end_date = datetime(2023, 9, 30)

# Define the time step (in hours, minutes, seconds, etc.)
time_step = timedelta(minutes=1)  # Adjust as needed

# Generate the datetime array
datetime_array = np.arange(start_date, end_date, time_step).astype(datetime)

length_vid = 120 # sec

fps = len(datetime_array)/length_vid



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

time_array = []
for idx in above_10.index:
    # Define start time and duration
    start_time = above_10.start[idx]  # May 1st, 2024, 00:00
    duration = timedelta(minutes=above_10.time_above[idx])  # Duration of 2 hours

    # Calculate end time
    end_time = start_time + duration

    # Generate time array
    time_array.append(np.arange(start_time, end_time, timedelta(minutes=1)).astype(datetime))


above_10['time'] = time_array
above_10 = above_10.explode('time')


#%%
'''# Set up the plot
fig, ax = plt.subplots()
sc = ax.scatter([], [])

# Function to update the plot for each frame
def update(frame):
    print(frame/len(datetime_array))
    ax.clear()
    t = above_10[above_10.time==datetime_array[frame]]

    ax.scatter(t.mean_lon,t.mean_lat, s=t.size, facecolors='none', edgecolors='blue', linewidths=.5)
    ax.set_xlim(250, 256)
    ax.set_ylim(37,41)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Timestep {frame + 1}/{len(datetime_array)}')
    return sc,

# Create animation
ani = FuncAnimation(fig, update, frames=len(datetime_array), blit=True)

# Save animation as video
ani.save('lat_lon_animation.mp4', fps=fps)  # Adjust fps as needed

plt.show()'''

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
    print(f"Frame {frame + 1}/{len(datetime_array)}: {frame / len(datetime_array):.2f}")
    t = above_10[above_10.time == datetime_array[frame]]
    
    if not t.empty:
        offsets = np.column_stack((t['mean_lon'].values, t['mean_lat'].values))
        sc.set_offsets(offsets)
        sc.set_sizes(np.ravel(t['size'].values))
        sc.set_edgecolor('red')
        sc.set_facecolor('none')
        sc.set_linewidths(0.5)
    else:
        sc.set_offsets(np.zeros((0, 2)))   # Clear scatter if there's no data for the current frame
        sc.set_sizes(np.array([]))
    
    ax.set_title(f'Timestep {frame + 1}/{len(datetime_array)}')
    return sc,

# Create animation
ani = FuncAnimation(fig, update, frames=len(datetime_array), init_func=init,blit=True)

length_vid = 60*20 # sec

fps = int(len(datetime_array)/length_vid)

ani.save('lat_lon_animation.mp4', fps=fps)

plt.show()