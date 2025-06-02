#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde

#%%
df_atlas = pd.read_feather('../output/atlas_14')
df_reps = pd.read_feather('../output/reps').drop(columns=['band_x', 'spatial_ref_x', 'band_y', 'spatial_ref_y'])
df_conus = pd.read_feather('../output/conus_new_ann_max').groupby(['latitude','longitude','year']).max().drop(columns='season').groupby(['latitude','longitude']).quantile(.9).reset_index()
df_conus['dataset'] = 'conus'
df = pd.concat([df_atlas,df_reps,df_conus])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])


#%%
xarray_list = []
for dataset in ['conus','refs','atlas14']:
    xarray_list.append(df[df.dataset==dataset].groupby(['latitude','longitude']).max().accum_1hr.to_xarray())

for dataset in ['conus','refs','atlas14']:
    xarray_list.append(df[df.dataset==dataset].groupby(['latitude','longitude']).max().accum_24hr.to_xarray())


#%%
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
custom_cmap.set_over("white") 
boundaries = np.arange(0,45,5)  # Boundaries for color changes
norm = BoundaryNorm(boundaries, custom_cmap.N)
boundaries = np.arange(0,90,10)  # Boundaries for color changes
norm2 = BoundaryNorm(boundaries, custom_cmap.N)
# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(14 * 0.7, 8 * 0.6), sharex=True, sharey=True)
elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude', 'longitude']).max().elevation.to_xarray()
shapefiles = {}
for shape in [10, 11, 13, 14]:
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    shapefiles[shape] = gpd.read_file(shapefile_path)
gdf2 = gpd.read_file("../data/Colorado_Mtn_Ranges/ranges.shp")
for i, ax in enumerate(axes.flat):
    data = xarray_list[i]
    #data = data.where(data > 0.25)
    ax.text(-104.4, 39.6, "10", 
            fontsize=14, color='black', ha='center')
    
    ax.text(-104.7, 37.1, "11", 
            fontsize=14, color='black', ha='center')
    ax.text(-106, 37.4, "13", 
            fontsize=14, color='black', ha='center')
    ax.text(-108.6, 39, "14", 
            fontsize=14, color='black', ha='center')
    # Plot shapefiles

    # Plot precipitation data with different vmax for each row
    if i < 3:
        im = ax.pcolormesh(data.longitude, data.latitude, data.values,cmap=custom_cmap,norm=norm, rasterized=True)
    else:
        im = ax.pcolormesh(data.longitude, data.latitude, data.values, cmap=custom_cmap, norm=norm2,rasterized=True)

    gdf2.plot(ax=ax, edgecolor='red', facecolor='none',linewidth=.75,linestyle='--')
    for gdf in shapefiles.values():
        gdf.plot(ax=ax, edgecolor='black', facecolor='none',linewidth=.5)
    ax.set_aspect('equal')
    ax.set_xlim(elev.longitude.min(), elev.longitude.max())
    ax.set_ylim(elev.latitude.min(), elev.latitude.max())

# Add row and column labels
row_labels = ['1-hr', '24-hr']
for row in range(2):
    fig.text(0, 0.75 - row * 0.47, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=14)

col_labels = ['CONUS404', 'REPS', 'ATLAS 14']
for col in range(3):
    fig.text(0.18 + col * 0.28, 1.0, col_labels[col], va='center', ha='center', rotation='horizontal', fontsize=12)

# Add a colorbar for each row
for row in range(2):
    cbar_ax = fig.add_axes([0.9, 0.6 - row * 0.47, 0.015, 0.35])  # Positioning for colorbar
    

    if row==0: 
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
                        cax=cbar_ax,extend='max')                   
        cbar.set_label('1hr accum, mm', fontsize=10)
    else:
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm2),
                        cax=cbar_ax,extend='max')
        cbar.set_label('24hr accum, mm', fontsize=10)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbars
plt.show()
fig.savefig("../figures_output/f06.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')


# %%
