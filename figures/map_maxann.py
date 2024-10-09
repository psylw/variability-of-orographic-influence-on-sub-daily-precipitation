#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import xesmf as xe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec
#%%
xarray_list = []

for window in (1,3,12,24):
    
    df_aorc = xr.open_dataset('../output/'+'aorc'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc').median(dim='year')

    df_nldas = xr.open_dataset('../output/'+'nldas'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc').median(dim='year')
    
    df_conus = xr.open_dataset('../output/'+'conus'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc').median(dim='year')

    #df_mrms = xr.open_dataset('../output/'+'mrms_nldas'+'_ann_max_px_2016'+'_window_'+str(window)+'.nc').median(dim='year').drop_vars(['step','heightAboveSea'])

    combined = xr.concat([df_conus, df_aorc, df_nldas], dim='ensemble')

    std_dev = combined.std(dim='ensemble')

    xarray_list.append(df_aorc)
    xarray_list.append(df_nldas)
    xarray_list.append(df_conus)
    #xarray_list.append(df_mrms)
    xarray_list.append(std_dev)


#%%
elev = pd.read_feather('../output/'+'nldas'+'_elev')
elev = elev.groupby(['latitude','longitude']).max().elevation.to_xarray()
#%%
# Create a 4x4 subplot

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
# Adjust the figure to make room for colorbars in the last two columns
fig, axes = plt.subplots(4, 4, figsize=(15 * 0.5, 15 * 0.5), sharex=True, sharey=True)



row_labels = ['1-hr', '3-hr', '12-hr', '24-hr']

vmax = [20, 20, 20,  10,
        25/3, 25/3, 25/3,  10/3,

        30/12, 30/12, 30/12,  10/12,
        35/24, 35/24, 35/24,  10/24]
vmin = [3, 3, 3,  0,
        4/3, 4/3, 4/3,  0,
        5/12, 5/12, 5/12,  0,
        6/24, 6/24, 6/24,  0]
title1 = 'med int, mm/hr'
title2 = 'std dev, mm/hr'
# Loop through each xarray and corresponding subplot
for i, ax in enumerate(axes.flat):
    ax.contour(elev.longitude, elev.latitude, elev.values, levels=5, alpha=0.5, colors='white')
    # Plot each xarray in a subplot
    # Replace 'your_variable' with the variable you want to plot
    im = xarray_list[i].accum.plot(ax=ax, vmin=vmin[i], vmax=vmax[i], add_colorbar=False, edgecolor=None, rasterized=True)
    if (i % 4) == 2 and (i // 4) == 0:
            cbar_ax_4th = fig.add_axes([.99, 0.76, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title1)
    if (i % 4) == 3 and (i // 4) == 0:
            cbar_ax_4th = fig.add_axes([1.07, 0.76, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title2)
    if (i % 4) == 2 and (i // 4) == 1:
            cbar_ax_4th = fig.add_axes([.99, 0.76-.24, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title1)
    if (i % 4) == 3 and (i // 4) == 1:
            cbar_ax_4th = fig.add_axes([1.07, 0.76-.24, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title2)
    if (i % 4) == 2 and (i // 4) == 2:
            cbar_ax_4th = fig.add_axes([.99, 0.76-.24*2, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title1)
    if (i % 4) == 3 and (i // 4) == 2:
            cbar_ax_4th = fig.add_axes([1.07, 0.76-.24*2, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title2)
    if (i % 4) == 2 and (i // 4) == 3:
            cbar_ax_4th = fig.add_axes([.99, 0.77-.24*3, 0.01, 0.19])  # Adjust the position as needed
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title1)
    if (i % 4) == 3 and (i // 4) == 3:
            cbar_ax_4th = fig.add_axes([1.07, 0.77-.24*3, 0.01, 0.19])  # Adjust the position as needed
            
            cbar = fig.colorbar(im, cax=cbar_ax_4th)
            cbar.set_label(title2)
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

# Add row labels
for row in range(4):
    fig.text(-0.01, 0.85 - row * 0.24, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=12)

# Adjust layout to avoid overlapping plots
plt.tight_layout()  # Adjust right spacing to make room for the last column

# Show the plot
plt.show()

fig.savefig("../figures_output/map_ann_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
