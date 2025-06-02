#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde

#%%
#df = pd.read_feather('../output/conus_ann_max')

df = pd.read_feather('../output/conus_new_ann_max')

df = df.groupby(['latitude','longitude','season']).quantile(.5).reset_index()

df_elev = pd.read_feather('../output/conus_elev')
df = pd.merge(df,df_elev,on=['latitude','longitude'])
# %%
window = 'accum_1hr'
df1 = df.copy()
df_max = df1[['huc2',window,'season']].groupby(['huc2','season']).max().rename(columns={window:'n_max'})

df_min = df1[['huc2',window,'season']].groupby(['huc2','season']).min().rename(columns={window:'n_min'})

df1 = pd.merge(df1,df_max,on=['huc2','season'])
df1 = pd.merge(df1,df_min,on=['huc2','season'])

df1['norm_1'] = (df1[window]-df1.n_min)/(df1.n_max-df1.n_min)

window = 'accum_24hr'
df2 = df.copy()
df_max = df2[['huc2',window,'season']].groupby(['huc2','season']).max().rename(columns={window:'n_max'})

df_min = df2[['huc2',window,'season']].groupby(['huc2','season']).min().rename(columns={window:'n_min'})

df2 = pd.merge(df2,df_max,on=['huc2','season'])
df2 = pd.merge(df2,df_min,on=['huc2','season'])

df2['norm_2'] = (df2[window]-df2.n_min)/(df2.n_max-df2.n_min)

#%%
for h in [10,11,13,14]:
    plot = df1[df1.huc2==h]
    sns.kdeplot(data=plot,x='norm_1',hue='season')
    plot = df2[df2.huc2==h]
    sns.kdeplot(data=plot,x='norm_2',hue='season',linestyle='--')
    plt.title(h)
    plt.show()
#%%
xarray_list = []
for season in ['DJF','MAM','JJA','SON']:
    xarray_list.append(df1[df1.season==season].groupby(['latitude','longitude']).max().norm_1.to_xarray())

for season in ['DJF','MAM','JJA','SON']:
    xarray_list.append(df2[df2.season==season].groupby(['latitude','longitude']).max().norm_2.to_xarray())


#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr

plot_thr = 0

cmap = plt.get_cmap('viridis')
boundaries = np.arange(plot_thr,1.25,.25)  # Boundaries for color changes
norm = BoundaryNorm(boundaries, cmap.N)


# Load shapefiles once
shapefiles = {}
for shape in [10, 11, 13, 14]:
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    shapefiles[shape] = gpd.read_file(shapefile_path)
gdf2 = gpd.read_file("../data/Colorado_Mtn_Ranges/ranges.shp")
# Create subplots
fig, axes = plt.subplots(2, 4, figsize=(13.5,5), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    data = xarray_list[i]
    #data = data.where(data > plot_thr)

    ax.text(-104.4, 39.6, "10", 
            fontsize=14, color='white', ha='center')
    ax.text(-104.7, 37.1, "11", 
            fontsize=14, color='white', ha='center')
    ax.text(-106, 37.4, "13", 
            fontsize=14, color='white', ha='center')
    ax.text(-108.6, 40.5, "14", 
            fontsize=14, color='white', ha='center')
    # Plot precipitation data
    im = ax.pcolormesh(data.longitude, data.latitude, data.values, cmap=cmap, norm=norm, rasterized=True)
    gdf2.plot(ax=ax, facecolor='none',linewidth=.75,linestyle='--', edgecolor='white')
    # Plot shapefiles
    for gdf in shapefiles.values():
        gdf.plot(ax=ax, edgecolor='red', facecolor='none')

    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(data.longitude.min(), data.longitude.max())
    ax.set_ylim(data.latitude.min(), data.latitude.max())

    #snotel = pd.read_csv('../data/snotel_loc.csv')
    #ax.scatter(snotel.Longitude,snotel.Latitude,c='white')
# Add row and column labels
row_labels = ['1-hr', '24-hr']
for row in range(2):
    fig.text(.01
    , 0.75 - row * 0.45, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=14)

col_labels = ['DJF','MAM','JJA','SON']
for col in range(4):
    fig.text(0.15 + col * 0.21, 1.0, col_labels[col], va='center', ha='center', rotation='horizontal', fontsize=12)

# Add a shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=-0.18)
cbar.set_label('Normalized Accumulation', fontsize=12)


plt.tight_layout()
#plt.subplots_adjust(top=0.9)  # Adjust top margin to fit labels
plt.show()

fig.savefig("../figures_output/f04.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
