#%%
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import cv2
from metpy.plots import USCOUNTIES
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# open storms above 20 mm/hr
ds = xr.open_dataset('../storms_test.nc')
df = pd.read_feather('../highint_test')

# %%
# Initialize video writer
video_name = 'xarray_video.avi'
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_writer = cv2.VideoWriter(video_name, fourcc, fps, (int(14*0.9*100), int(10*0.9*100)))

fig = plt.figure(1, figsize=(14*0.9,10*0.9))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                        hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
ax = plt.subplot(1,1,1, projection=plotcrs)

# Plot counties and states only once
ax.add_feature(cfeature.STATES, linewidth=1)
ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

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

# Prepare colorbar
elev = ax.contourf(ds.longitude, ds.latitude, ds.isel(time=0).intensity, levels=[0, 20, 30, 40, 50], origin='upper', cmap='terrain', 
                    alpha=0.4, transform=ccrs.PlateCarree())
cb = fig.colorbar(elev, orientation="horizontal", shrink=.55, pad=0.01)
cb.ax.tick_params(labelsize=12)
cb.set_label("elevation (m)", fontsize=12)

for i in range(0,500):
    # Update contour plot for each time step
    for coll in elev.collections:
        coll.remove()  # Remove existing contours
    elev = ax.contourf(ds.longitude, ds.latitude, ds.isel(time=i).intensity, levels=[0, 20, 30, 40, 50], origin='upper', cmap='terrain', 
                        alpha=0.4, transform=ccrs.PlateCarree())
    timestamp = ds.time[i].values
    ax.set_title(f'Timestep {i} - {timestamp}', fontsize=14)
    
    # Draw the figure on the canvas
    fig.canvas.draw()

    # Convert the figure to a numpy array
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Resize the frame to fit the video dimensions
    frame = cv2.resize(frame, (int(14*0.9*100), int(10*0.9*100)))
    
    # Write the frame to the video file
    video_writer.write(frame)

# Release the video writer
video_writer.release()

plt.close(fig)
# %%
