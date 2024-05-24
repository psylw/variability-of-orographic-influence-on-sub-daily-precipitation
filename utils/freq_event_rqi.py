#%%
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd

# open data
rqi_folder = '..\\..\\data\\MRMS\\RQI_2min_cat_month_CO\\'
file_rqi = glob.glob(rqi_folder+'*.grib2')[-15:]

threshold = .8

#%%
df = pd.read_feather('..\\output\\count_duration_threshold_px2')
df = df[df.unknown>40]
df['year'] = [df.time[i].year for i in df.index]
# %%
da = df.groupby(['latitude','longitude']).max().unknown.to_xarray()

#%%
# find events where precip > 40 mm/hr and rqi > .8
rqi_above_th = []
for file in file_rqi:
    print(file)
    rqi = xr.open_dataset(file)
    rqi = rqi.where(rqi.unknown>threshold)
    rqi = rqi.sel(latitude = da.latitude, longitude = da.longitude)
    test = rqi.where(rqi.time.isin(df.time),drop=True)
    test = test.to_dataframe().unknown
    test = test.dropna().reset_index()
    test = test.rename(columns={'unknown':'rqi'})
    df = df.merge(test, on=['time','latitude','longitude'],how='left')
#%%
df['rqi'] = df.iloc[:,-15:].max(1)
df=df.drop(columns=['rqi_x', 'rqi_y', 'rqi_x',
       'rqi_y', 'rqi_x', 'rqi_y', 'rqi_x', 'rqi_y', 'rqi_x', 'rqi_y', 'rqi_x',
       'rqi_y', 'rqi_x', 'rqi_y'])
df = df.dropna()
#%%
# get total time > .8
file_area = glob.glob('month'+'*')
area = []
for file in file_area:
    area.append(pd.read_feather(file))
area = pd.concat(area)
area = area.groupby(['latitude','longitude']).sum() # minimum monthly timesteps above 0.8

# %%
freq_day = df.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))/(area.unknown*2*(1/1440))

freq_day = freq_day.dropna().reset_index().groupby(['latitude','longitude']).max().to_xarray()[0]

#%%
df = pd.read_feather('..\\output\\count_duration_threshold_px2')
df = df[df.unknown>40]
df['year'] = [df.time[i].year for i in df.index]
total_storms = df[df.year>2020].reset_index()
total_storms = total_storms[total_storms.unknown>40]
unique_values = total_storms.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))

freq_day_no_rqi = unique_values.to_xarray()/459
(freq_day_no_rqi).plot()
# %%
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
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
fig,axs = plt.subplots(1,1, subplot_kw=dict(projection=plotcrs), figsize=(12,8))

elev=plt.contourf(freq_day_no_rqi.longitude,freq_day_no_rqi.latitude,freq_day_no_rqi, origin='upper',
            alpha=0.6,transform=ccrs.PlateCarree(),levels=list(np.arange(.01,.06,.01)),extend='both')

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
# %%
