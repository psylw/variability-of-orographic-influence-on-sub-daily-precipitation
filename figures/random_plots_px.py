#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import timedelta

grouped = pd.read_feather('..\\output\\count_duration_threshold_px2')
grouped['year'] = [grouped.time[i].year for i in grouped.index]
# %%
# plot max intensity at each pixel
total_storms = grouped[grouped.year>2020]
all_years = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
plt.show()

for year in range(2021,2024):
    total_storms = grouped[grouped.year==year]
    total_storms = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
    plt.title(year)
    plt.show()
# %%
# max above threshold duration
total_storms = grouped[grouped.year>2020]
total_storms = total_storms[total_storms.unknown>40]
total_storms = total_storms.groupby(['latitude','longitude','storm_id']).count().reset_index()

total_storms = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
plt.title('all years')
plt.show()

for year in range(2021,2024):
    total_storms = grouped[grouped.year==year]
    total_storms = total_storms.groupby(['latitude','longitude','storm_id']).count().reset_index()

    total_storms = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
    plt.title(year)
    plt.show()

# %%
# above threshold frequency
total_storms = grouped[grouped.year>2020].reset_index()
total_storms = total_storms[total_storms.unknown>40]
unique_values = total_storms.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))
unique_values.to_xarray().plot()
#%%
for year in range(2021,2024):
    total_storms = grouped[grouped.year==year]
    #total_storms = total_storms[total_storms.unknown>40]
    unique_values = total_storms.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))
    unique_values.to_xarray().plot()
    plt.title(year)
    plt.show()
#%%
times_perloc_perstorm = total_storms.groupby(['latitude','longitude','storm_id']).count().time.reset_index()
times_perloc_perstorm = times_perloc_perstorm.rename(columns={'time':'duration'})
total_storms = pd.merge(total_storms,times_perloc_perstorm,on=['latitude','longitude','storm_id'])
#%%
med_dur = times_perloc_perstorm.duration.quantile(.5)
med_int = total_storms.unknown.quantile(.9)

#%%
above_dur = total_storms.loc[(total_storms.duration>1)&(total_storms.unknown>40)]

unique_values = above_dur.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))
unique_values.to_xarray().plot()
plt.title('all years')
plt.show()

############################################################################################################
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import geopandas as gpd
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature
import cartopy.crs as ccrs

import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import xarray as xr
import os
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

plt.rcParams['figure.dpi'] = 150
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


#%%
# map of gage
plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)

fig,axs = plt.subplots(1,3, subplot_kw=dict(projection=plotcrs), figsize=(8,4.5))

gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                    hspace=0.01, wspace=0.01)
idx=0
for year in range(2021,2024):
###################
#for i in range(3):
    
       
    total_storms = grouped.loc[(grouped.year==year)&(grouped.unknown>40)].reset_index()

    #unique_values = total_storms.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))
    total_storms = total_storms.groupby(['latitude','longitude','storm_id']).count().reset_index()
    unique_values = total_storms.groupby(['latitude','longitude']).max().unknown

    df_th_count = unique_values.to_xarray()
    df_th_count = df_th_count.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
    plot_map = df_th_count
    name_cb = 'minutes'
    y,x = df_th_count.latitude,df_th_count.longitude
    #levels = list(np.arange(0,20,3))
    levels = list(np.arange(0,80,10))
    plt.suptitle('above threshold max duration (minutes)', y=0.77)
    #plt.suptitle('above threshold frequency (events/year)', y=0.77)


    axs[idx].set_extent((-109.2, -103.8, 36.9, 41.1))

    elev=axs[idx].contourf(x,y,plot_map, cmap=cmap2,origin='upper', transform=ccrs.PlateCarree(),extend='both'
                        ,levels=levels)
    axs[idx].set_title(str(year), y=-0.12)
    elev2=axs[idx].contour(noband.longitude,noband.latitude,noband.values, origin='upper', transform=ccrs.PlateCarree(),colors='grey',levels=list(np.arange(2500,4000,500)),linewidths=.25)

    axs[idx].add_feature(cfeature.STATES, linewidth=.5)


    if idx==0:
        gl = axs[idx].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
    else:
        gl = axs[idx].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
        gl.left_labels=False # suppress right labels
    idx+=1

''''''
fig.tight_layout()
fig.subplots_adjust(right=.85)
cbar_ax = fig.add_axes([.88, 0.25, 0.02, 0.5])
cb=fig.colorbar(elev, cax=cbar_ax)
cb.ax.tick_params(labelsize=8)
cb.set_label(name_cb, fontsize=12)

#plt.show()


# %%
