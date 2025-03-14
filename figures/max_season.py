#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde


#%%

df = pd.read_feather('../output/conus_new_ann_max')
t1 = df.drop(columns='index').groupby(['latitude','longitude','year','season']).max().to_xarray()
t2 = pd.read_feather('../output/conus_new_ann_max').groupby(['latitude','longitude','year']).max().drop(columns='season').groupby(['latitude','longitude']).quantile(.9).drop(columns='index').to_xarray()

df1 = t1.where(t1.accum_1hr>=t2.accum_1hr).to_dataframe().dropna()['accum_1hr'].reset_index()
df2 = t1.where(t1.accum_24hr>=t2.accum_24hr).to_dataframe().dropna()['accum_24hr'].reset_index()
#%%

idx1 = df1.groupby(['latitude', 'longitude', 'year'])['accum_1hr'].idxmax()
max_accum_df1 = df1.loc[idx1]
most_common_season_per_location1 = (
    max_accum_df1
    .groupby(['latitude', 'longitude'])['season']
    .apply(lambda x: x.value_counts().idxmax())
    .reset_index(name='most_common_season')
)

idx2 = df2.groupby(['latitude', 'longitude', 'year'])['accum_24hr'].idxmax()
max_accum_df2 = df2.loc[idx2]
most_common_season_per_location2 = (
    max_accum_df2
    .groupby(['latitude', 'longitude'])['season']
    .apply(lambda x: x.value_counts().idxmax())
    .reset_index(name='most_common_season')
)

season_map = {
    'DJF': 1,
    'MAM': 2,
    'JJA': 3,
    'SON': 4
}

# Replace existing "season" column with its integer equivalent
most_common_season_per_location1['season_int'] = most_common_season_per_location1['most_common_season'].map(season_map)
most_common_season_per_location2['season_int'] = most_common_season_per_location2['most_common_season'].map(season_map)
#%%
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm

# 1) Read shapefiles (example loop)
shapefiles = {}
for shape in [10, 11, 13, 14]:
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    shapefiles[shape] = gpd.read_file(shapefile_path)

gdf2 = gpd.read_file("../data/Colorado_Mtn_Ranges/ranges.shp")

# 2) Prepare your 2 xarray datasets (ds1 and ds2) from grouped DataFrames
ds1 = (
    most_common_season_per_location1
    .groupby(["latitude", "longitude"])
    .max()
    .to_xarray()
)

ds2 = (
    most_common_season_per_location2
    .groupby(["latitude", "longitude"])
    .max()
    .to_xarray()
)

# 3) Create figure and subplots
fig, (ax1, ax2) = plt.subplots(
    1, 2,                # 1 row, 2 columns
    figsize=(12, 4),     # adjust as desired
    subplot_kw={'projection': None}, sharex=True, sharey=True  # if you're using a map projection, set it here
)

# 4) Define a discrete colormap and matching norm
cmap = plt.get_cmap('YlGnBu')           # or define a ListedColormap
bounds = np.arange(4 + 1) + 0.5         # e.g. [0.5, 1.5, 2.5, 3.5, 4.5] for 4 seasons
norm = BoundaryNorm(bounds, cmap.N)

# 5) Plot each dataset to its respective axis, disabling auto-colorbar
plot_obj1 = ds1.season_int.plot(
    ax=ax1,
    cmap=cmap,
    norm=norm,
    add_colorbar=False, rasterized=True
)

plot_obj2 = ds2.season_int.plot(
    ax=ax2,
    cmap=cmap,
    norm=norm,
    add_colorbar=False, rasterized=True
)

# 6) Overlay shapefiles on each axis
for gdf in shapefiles.values():
    gdf.plot(ax=ax1, edgecolor='black', facecolor='none')
    gdf.plot(ax=ax2, edgecolor='black', facecolor='none')

gdf2.plot(ax=ax1, edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--')
gdf2.plot(ax=ax2, edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--')

# 7) Create a single ScalarMappable for the colorbar
#    This ensures one colorbar can be used for both plots.
from matplotlib.cm import ScalarMappable

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # dummy array to associate with the colorbar

cbar = fig.colorbar(
    plot_obj1,
    ax=[ax1,ax2],        # both subplots share the colorbar
        # or 'bottom'
    pad=-.45,             # space between subplot and colorbar
    shrink=0.8            # adjust size
)

# 8) Customize the colorbar labels
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels(["DJF", "MAM", "JJA", "SON"])

# 9) Titles and layout
ax1.set_title("1-hr")
ax2.set_title("24-hr")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax2.set_xlabel("")
ax2.set_ylabel("")

plt.tight_layout()
plt.show()

fig.savefig("../figures_output/season_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')