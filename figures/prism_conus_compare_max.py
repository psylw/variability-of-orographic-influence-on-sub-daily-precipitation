#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd

p = xr.open_dataset('../output/prism_season_max.nc')
c = pd.read_feather('../output/conus_new_ann_max')
#%%
c['year_season'] = c.year.astype('str')+'_'+c.season

c = c.groupby(['latitude','longitude','year_season']).max().accum_24hr.to_xarray()

# %%
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm

# Example discrete levels for the difference
levels_diff = [-15,-10,-5,0, 5,10,15]
cmap_diff = plt.get_cmap('coolwarm')
norm_diff = BoundaryNorm(levels_diff, cmap_diff.N)

cmap_diff.set_under("white")  # color for values below vmin
cmap_diff.set_over("black") 

# Example discrete levels for the mean (adjust to your data range)
levels_son = np.arange(0,.5+.125,.125)  # <-- Adjust as needed
cmap_son = plt.get_cmap('YlGnBu')
norm_son = BoundaryNorm(levels_son, cmap_son.N)
cmap_son.set_over("white")

# Read shapefiles (example)
shapefiles = {}
for shape in [10, 11, 13, 14]:
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    shapefiles[shape] = gpd.read_file(shapefile_path)
gdf2 = gpd.read_file("../data/Colorado_Mtn_Ranges/ranges.shp")

# Prepare subplots: 2 rows x 4 columns
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17, 8), sharex=True, sharey=True)

seasons = ['DJF', 'MAM', 'JJA', 'SON']
for i, season in enumerate(seasons):

    #
    # --- First row (0): Difference Plot ---
    #
    ax_diff = axes[0, i]

    # Mask data for season
    mask_c = c.year_season.str.endswith(season)
    ds_son = c.sel(year_season=c.year_season[mask_c])

    mask_p = p.year_season.str.endswith(season)
    ds_son_p = p.sel(year_season=p.year_season[mask_p])

    # Compute difference
    
    diff = ds_son.quantile(.5,dim='year_season') - ds_son_p.band_data.quantile(.5,dim='year_season')

    # Plot the difference
    im_diff = diff.plot(
        ax=ax_diff,
        cmap=cmap_diff,
        norm=norm_diff,
        add_colorbar=False, rasterized=True   # We'll make a single colorbar for the top row afterward
    )

    # Plot shapefiles and text
    gdf2.plot(ax=ax_diff, edgecolor='aqua', facecolor='none',linestyle='--')
    for shape_id, gdf_shp in shapefiles.items():
        gdf_shp.plot(ax=ax_diff, edgecolor='black', facecolor='none')
    ax_diff.text(-104.4, 39.6, "10", fontsize=14, color='black', ha='center')
    ax_diff.text(-104.7, 37.1, "11", fontsize=14, color='black', ha='center')
    ax_diff.text(-106,   37.4, "13", fontsize=14, color='black', ha='center')
    ax_diff.text(-108.6, 40.5, "14", fontsize=14, color='black', ha='center')

    ax_diff.set_title(season, fontsize=14)
    ax_diff.set_xlim(diff.longitude.min(), diff.longitude.max())
    ax_diff.set_ylim(diff.latitude.min(), diff.latitude.max())
    ax_diff.set_xlabel('')
    ax_diff.set_ylabel('')

    #
    # --- Second row (1): Mean Plot (ds_son) ---
    #
    ax_son = axes[1, i]
    
    # Plot the mean of ds_son
    # (Using the same ds_son defined above)
    im_son = abs((diff/ds_son.quantile(.5,dim='year_season'))).plot(
        ax=ax_son,
        cmap=cmap_son,
        norm=norm_son,
        add_colorbar=False, rasterized=True  # We'll create one colorbar for this row
    )
    ax_son.set_title(None)
    # Plot shapefiles & text again
    gdf2.plot(ax=ax_son, edgecolor='red', facecolor='none',linestyle='--')
    for shape_id, gdf_shp in shapefiles.items():
        gdf_shp.plot(ax=ax_son, edgecolor='black', facecolor='none')
    ax_son.text(-104.4, 39.6, "10", fontsize=14, color='white', ha='center')
    ax_son.text(-104.7, 37.1, "11", fontsize=14, color='white', ha='center')
    ax_son.text(-106,   37.4, "13", fontsize=14, color='white', ha='center')
    ax_son.text(-108.6, 40.5, "14", fontsize=14, color='white', ha='center')

    #ax_son.set_title(f"{season} (Mean)", fontsize=14)
    ax_son.set_xlim(diff.longitude.min(), diff.longitude.max())
    ax_son.set_ylim(diff.latitude.min(), diff.latitude.max())
    ax_son.set_xlabel('')
    ax_son.set_ylabel('')
    
    #snotel = pd.read_csv('../data/snotel_loc.csv')
    #ax_diff.scatter(snotel.Longitude,snotel.Latitude, facecolor='none',edgecolors='white',linewidth=1, s=20)

# Create colorbar for the top row (difference)
cbar_diff = fig.colorbar(
    im_diff,
    ax=axes[0, :],          # All columns in row 0
    orientation='vertical',
    fraction=0.02,
    pad=-.18,
    extend='both'
)
cbar_diff.set_label("CONUS404 - PRISM (mm)")

# Create colorbar for the bottom row (mean)
cbar_son = fig.colorbar(
    im_son,
    ax=axes[1, :],          # All columns in row 1
    orientation='vertical',
    extend='max',
    fraction=0.02,
    pad=-.18,

    
)
cbar_son.set_label("MRE")

plt.tight_layout()
plt.show()

fig.savefig("../figures_output/prism_compare_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
