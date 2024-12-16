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
df_mrms = pd.read_feather('../output/mrms_ann_max').drop(columns=['step','heightAboveSea'])
df_mrms = df_mrms.groupby(['latitude','longitude']).quantile(.5).reset_index()
df_mrms['dataset'] = 'mrms'
df_aorc = pd.read_feather('../output/aorc_ann_max')
df_aorc = df_aorc.groupby(['latitude','longitude']).quantile(.5).reset_index()
df_aorc['dataset'] = 'aorc'

df_conus = pd.read_feather('../output/conus_ann_max')
df_conus = df_conus[(df_conus.season=='JJA')&(df_conus.year>=2016)].drop(columns='season')
df_conus = df_conus.groupby(['latitude','longitude']).quantile(.5).reset_index()
df_conus['dataset'] = 'conus'
df = pd.concat([df_mrms,df_aorc,df_conus])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])
#%%
#%%
xarray_list = []
for dataset in ['conus','mrms','aorc']:
    xarray_list.append(df[df.dataset==dataset][['latitude','longitude','accum_1hr']].groupby(['latitude','longitude']).median().accum_1hr.to_xarray())

for dataset in ['conus','mrms','aorc']:
    xarray_list.append(df[df.dataset==dataset][['latitude','longitude','accum_24hr']].groupby(['latitude','longitude']).median().accum_24hr.to_xarray())


# Create a 4x4 subplot
#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from matplotlib.patches import Rectangle
# Define the color palette
colors = [
    "#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb",
    "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"
]

# Create the custom colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(14 * 0.7, 8 * 0.6), sharex=True, sharey=True)
elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude', 'longitude']).max().elevation.to_xarray()
shapefiles = {}
for shape in [10, 11, 13, 14]:
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    shapefiles[shape] = gpd.read_file(shapefile_path)

for i, ax in enumerate(axes.flat):
    data = xarray_list[i]
    data = data.where(data > 0.25)
    # Plot shapefiles

    # Plot precipitation data with different vmax for each row
    if i < 3:
        im = ax.pcolormesh(data.longitude, data.latitude, data.values, vmin=0, vmax=20, cmap=custom_cmap, rasterized=True)
    else:
        im = ax.pcolormesh(data.longitude, data.latitude, data.values, vmin=0, vmax=40, cmap=custom_cmap, rasterized=True)
    for gdf in shapefiles.values():
        gdf.plot(ax=ax, edgecolor='white', facecolor='none',linewidth=.5)
    ax.set_aspect('equal')
    ax.set_xlim(elev.longitude.min(), elev.longitude.max())
    ax.set_ylim(elev.latitude.min(), elev.latitude.max())

# Add row and column labels
row_labels = ['1-hr', '24-hr']
for row in range(2):
    fig.text(0, 0.75 - row * 0.47, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=14)

col_labels = ['CONUS404', 'MRMS', 'AORC']
for col in range(3):
    fig.text(0.18 + col * 0.28, 1.0, col_labels[col], va='center', ha='center', rotation='horizontal', fontsize=12)

# Add a colorbar for each row
for row in range(2):
    cbar_ax = fig.add_axes([0.9, 0.6 - row * 0.47, 0.015, 0.35])  # Positioning for colorbar
    vmax = 20 if row == 0 else 40
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax)),
                        cax=cbar_ax)
    if row==0:                    
        cbar.set_label('1hr accum, mm', fontsize=10)
    else:
        cbar.set_label('24hr accum, mm', fontsize=10)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbars
plt.show()
fig.savefig("../figures_output/med_annmax.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')


# %%
