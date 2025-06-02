#%%
import xarray as xr
import xesmf as xe
import glob
import matplotlib.pyplot as plt
files = glob.glob('../data/WWLLN_lightning/*.nc')


ds = xr.open_dataset('../data/WWLLN_lightning/WWLLN_th_2013.nc')
ds = (
    ds
    .set_coords(["lat", "lon", "mon"])        # flag as coords
    .swap_dims({"nlat": "lat", "nlon": "lon",         # replace indices
                "nmon": "mon"})                      # optional cleanup
)

df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016_JJA.nc')
df_conus = df_conus.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))

df = ds.sel(lon = slice(-109,-104.005),lat = slice(41,37))
regridder = xe.Regridder(df, df_conus, "conservative")


# %%
from pathlib import Path

ds_all = []
for file in sorted (files):
    ds = xr.open_dataset(file)


    ds = (
    ds
    .set_coords(["lat", "lon", "mon"])        # flag as coords
    .swap_dims({"nlat": "lat", "nlon": "lon",         # replace indices
                "nmon": "mon"})                      # optional cleanup
    )
    ds = ds.sel(lon = slice(-109,-104.005),lat = slice(41,37))
    ds = regridder(ds)

    ds = ds.where(ds.mon.isin([6,7,8]))
    ds = ds.sum(dim='mon')
    #ds = ds.max(dim='mon')

    year = int(Path(file).stem[-4::])
    ds = ds.expand_dims(year=[year])
    ds_all.append(ds)

ds_all = xr.concat(ds_all,dim='year').median(dim='year')


#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde

thunder = ds_all.to_dataframe().reset_index().rename(columns={'thunder_hours':'var'})
thunder['dataset'] = 'thunder'

df_conus = pd.read_feather('../output/conus_new_ann_max')
df_conus = df_conus[(df_conus.season=='JJA')].drop(columns='season')
df_conus = df_conus.groupby(['latitude','longitude']).median().reset_index()[['latitude', 'longitude', 'accum_24hr']].rename(columns={'accum_24hr':'var'})
df_conus['dataset'] = 'conus'
#%%
df = pd.concat([thunder,df_conus])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])


# %%

df2 = df.copy()
df_max = df2[['huc2','var','dataset']].groupby(['huc2','dataset']).max().rename(columns={'var':'n_max'})

df_min = df2[['huc2','var','dataset']].groupby(['huc2','dataset']).min().rename(columns={'var':'n_min'})

df2 = pd.merge(df2,df_max,on=['huc2','dataset'])
df2 = pd.merge(df2,df_min,on=['huc2','dataset'])

df2['norm'] = (df2['var']-df2.n_min)/(df2.n_max-df2.n_min)


#%%
xarray_list = []

for dataset in ['conus','thunder']:
    xarray_list.append(df2[df2.dataset==dataset][['latitude','longitude','norm']].groupby(['latitude','longitude']).median().norm.to_xarray())


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
fig, axes = plt.subplots(1, 2, figsize=(8,12), sharex=True, sharey=True)

titles = ['CONUS404','WWLLN']
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
    
    im = ax.pcolormesh(data.longitude, data.latitude, data.values, cmap=cmap, norm=norm, rasterized=True)
    # Plot shapefiles

    gdf2.plot(ax=ax, edgecolor='white', facecolor='none',linewidth=.75,linestyle='--')

    for gdf in shapefiles.values():
        gdf.plot(ax=ax, edgecolor='red', facecolor='none')

    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(data.longitude.min(), data.longitude.max())
    ax.set_ylim(data.latitude.min(), data.latitude.max())
    ax.set_title(titles[i])



# Add a shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=-0.18)
cbar.set_label('Normalized Values', fontsize=12)

plt.tight_layout()
#plt.subplots_adjust(top=0.9)  # Adjust top margin to fit labels
plt.show()

fig.savefig("../figures_output/f11.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')