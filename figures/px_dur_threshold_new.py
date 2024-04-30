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


df = pd.read_feather('../output/count_duration_threshold_px2')
df['year'] = [df.start_storm[i].year for i in df.index]

#df['year'] = [df.time[i].year for i in df.index]
#df=df[df.year>2020]

#%%
#df=df.loc[(df.latitude!=40.624999999999282)&(df.longitude!=253.33499899999782)]
shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)

threshold = [20,30,40]
for year in df.year.unique():
###################
#for i in range(3):
    # map of gage
    plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)

    fig,axs = plt.subplots(1,3, subplot_kw=dict(projection=plotcrs), figsize=(8,4.5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                        hspace=0.01, wspace=0.01)
    
    for idx,th in enumerate(threshold):
        print(year)
        #df_th = df.loc[(df.unknown >= th)&(df.year == year)]
           
        df_th = df.loc[(df.threshold == th)&(df.year == year)]
        # get median duration above threshold for each pixel
        df_th_median = df_th.groupby(['latitude','longitude']).median().dur_above_min.to_xarray()

        # get total duration above threshold for each pixel
        df_th_sum = (df_th.groupby(['latitude','longitude']).sum().dur_above_min/60).to_xarray()

        # get count of events above threshold for each pixel
        df_th_count = (df_th.groupby(['latitude','longitude']).count().dur_above_min/(9*5)).to_xarray()


        '''            if i==0:
            plot_map = df_th_median
            name_cb = 'minutes'
            y,x = df_th_median.latitude,df_th_median.longitude
            levels = list(np.arange(1,36,5))
            plt.suptitle('median duration above threshold', y=0.77)
        
        elif i==1:
            plot_map = df_th_sum
            name_cb = str(th)+' mm/hr'
            y,x = df_th_sum.latitude,df_th_sum.longitude
            levels = list(np.arange(1,31,5))
            plt.suptitle('total duration (hrs) above threshold', y=0.77)
        elif i ==2:'''
        df_th_count = (df_th.groupby(['latitude','longitude']).count().dur_above_min/(9*5)).to_xarray()
        plot_map = df_th_count
        name_cb = 'frequency'
        y,x = df_th_count.latitude,df_th_count.longitude
        levels = list(np.arange(0,1.25,.15))
        plt.suptitle('above threshold frequency (events/month)', y=0.77)


#################################################################
        # Set plot bounds -- or just comment this out if wanting to plot the full domain

        axs[idx].set_extent((-109.2, -103.8, 36.9, 41.1))

        elev=axs[idx].contourf(x,y,plot_map, cmap=cmap2,origin='upper', transform=ccrs.PlateCarree(),extend='both'
                            ,levels=levels)
        axs[idx].set_title(str(threshold[idx])+' mm/hr', y=-0.12)
        elev2=axs[idx].contour(noband.longitude,noband.latitude,noband.values, origin='upper', transform=ccrs.PlateCarree(),colors='grey',levels=list(np.arange(2500,4000,500)),linewidths=.25)



        #gdf.plot(ax = axs[idx],edgecolor='gray', linewidth=1.5,facecolor='none',transform=ccrs.PlateCarree())
        axs[idx].add_feature(cfeature.STATES, linewidth=.5)

        #axs[idx].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

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

    fig.tight_layout()
    fig.subplots_adjust(right=.85)
    cbar_ax = fig.add_axes([.88, 0.25, 0.02, 0.5])
    cb=fig.colorbar(elev, cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(name_cb, fontsize=12)

    plt.show()


# %%
