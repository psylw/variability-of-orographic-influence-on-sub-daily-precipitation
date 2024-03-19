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

plt.rcParams['figure.dpi'] = 150
#%%

df = pd.read_feather('../output/count_duration_threshold_px')

shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)

threshold = [10,20,30]
###################
for i in range(3):
    # map of gage
    plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)

    fig,axs = plt.subplots(1,3, subplot_kw=dict(projection=plotcrs), figsize=(12,5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                        hspace=0.01, wspace=0.01)
    for idx,th in enumerate(threshold):
        df_th = df.loc[df.threshold == th]

        # get median duration above threshold for each pixel
        df_th_median = df_th.groupby(['latitude','longitude']).median().dur_above_min.to_xarray()

        # get total duration above threshold for each pixel
        df_th_sum = (df_th.groupby(['latitude','longitude']).sum().dur_above_min/60).to_xarray()

        # get count of events above threshold for each pixel
        df_th_count = df_th.groupby(['latitude','longitude']).count().dur_above_min.to_xarray()

        if i==0:
            plot_map = df_th_median
            name_cb = 'median duration (min) above '+str(th)+' mm/hr'
            y,x = df_th_median.latitude,df_th_median.longitude
            levels = list(np.arange(0,35,5))

        if i==1:
            plot_map = df_th_sum
            name_cb = 'total duration (hrs) above '+str(th)+' mm/hr'
            y,x = df_th_sum.latitude,df_th_sum.longitude
            levels = list(np.arange(0,500,50))

        elif i ==2:
            plot_map = df_th_count
            name_cb = 'events above '+str(th)+' mm/hr'
            y,x = df_th_count.latitude,df_th_count.longitude
            levels = list(np.arange(0,1000,150))

#################################################################
        # Set plot bounds -- or just comment this out if wanting to plot the full domain
        axs[idx].set_extent((-109.2, -103.5, 36.8, 41.3))

        elev=axs[idx].contourf(x,y,plot_map, cmap=cmap2,origin='upper', transform=ccrs.PlateCarree(),extend='both'
                            ,levels=levels)

        cb =plt.colorbar(elev, orientation="horizontal",pad=0.01,ax=axs[idx])
        cb.ax.tick_params(labelsize=10)
        cb.set_label(name_cb, fontsize=12)

        gdf.plot(ax = axs[idx],edgecolor='red', linewidth=1,facecolor='none',transform=ccrs.PlateCarree())
        axs[idx].add_feature(cfeature.STATES, linewidth=.5)

        axs[idx].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')
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
    plt.show()

# %%
