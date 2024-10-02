#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xesmf as xe
from matplotlib import gridspec
#%%
xarray_list = []
for window in (1,3,12,24):
    
    df_aorc = xr.open_dataset('../output/'+'aorc'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc')

    df_nldas = xr.open_dataset('../output/'+'nldas'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc')
    
    df_conus = xr.open_dataset('../output/'+'conus'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc')

    combined = xr.concat([df_conus, df_aorc, df_nldas], dim='ensemble')

    std_dev = combined.std(dim='ensemble')

    xarray_list.append(df_aorc)
    xarray_list.append(df_nldas)
    xarray_list.append(df_conus)
    xarray_list.append(std_dev)
#%%
elev = pd.read_feather('../output/'+'nldas'+'_elev')
elev = elev.groupby(['latitude','longitude']).max().elevation.to_xarray()
#%%
# Create a 4x4 subplot

fig, axes = plt.subplots(4, 4, figsize=(15*.8, 12*.8), sharex=True, sharey=True)


row_labels = ['1-hr', '3-hr', '12-hr', '24-hr']

vmax = [20,20,20,4,
        25,25,25,5,
        30,30,30,6,
        35,35,35,7]
vmin = [3,3,3,0,
        4,4,4,0,
        5,5,5,0,
        6,6,6,0]

# Loop through each xarray and corresponding subplot
for i, ax in enumerate(axes.flat):
    ax.contour(elev.longitude,elev.latitude,elev.values, levels=5,alpha=.5, colors='white')
    # Plot each xarray in a subplot
    # Replace 'your_variable' with the variable you want to plot
    if (i % 4) < 2:  # First three columns
        im = xarray_list[i].accum.plot(ax=ax, vmin=vmin[i], vmax=vmax[i], add_colorbar=False)
    elif (i % 3) == 3:
        im = xarray_list[i].accum.plot(ax=ax, vmin=vmin[i], vmax=vmax[i], add_colorbar=False)

        fig.colorbar(im, ax=ax)
    else:  # Last column (individual colorbars)
        im = xarray_list[i].accum.plot(ax=ax, vmin=vmin[i], vmax=vmax[i], add_colorbar=False)
        cax = inset_axes(ax, width="5%", height="100%", loc='right', borderpad=-2)
        fig.colorbar(im, ax=ax, cax=cax)

    # Optional: Add a title to each subplot
    if i == 0:
        ax.set_title('AORC')
    if i == 1:
        ax.set_title('NLDAS-2')
    if i == 2:
        ax.set_title('CONUS404')
    if i == 3:
        ax.set_title('Std Dev')

    ax.set_xlabel('')
    ax.set_ylabel('')
for row in range(4):
    fig.text(-.01, 0.85 - row * 0.24, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=12)

plt.subplots_adjust(wspace=.5, hspace=0.1)
# Adjust layout to avoid overlapping plots
plt.tight_layout()

# Show the plot
plt.show()
fig.savefig("../figures_output/map_ann_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
