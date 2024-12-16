#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde
#%%
coag1 = pd.read_feather('../output/coag_1').drop(columns='id')
coag24 = pd.read_feather('../output/coag_24')

coag24['year'] = coag24.time.dt.year
coag1 = coag1.groupby(['latitude','longitude','year']).max().reset_index().groupby(['latitude','longitude']).median().reset_index()
coag24 = coag24.groupby(['latitude','longitude','year']).max().reset_index().groupby(['latitude','longitude']).median().reset_index()

coag_coord = coag1.groupby(['latitude','longitude']).max().reset_index()

elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude', 'longitude']).max().huc2.to_xarray()

huc = []
for coord in coag_coord.index:
    data = {'latitude':coag_coord.latitude[coord],
    'longitude':coag_coord.longitude[coord],
    'huc2':elev.sel(latitude = coag_coord.latitude[coord],longitude= coag_coord.longitude[coord],method='nearest').values}
    huc.append(data)

huc = pd.DataFrame(huc)
huc['huc2'] = huc.huc2.astype('float')

coag1 = pd.merge(coag1,huc,on=['latitude','longitude'])
coag24 = pd.merge(coag24,huc,on=['latitude','longitude'])

coag1['norm']=coag1.groupby('huc2')['accum'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
coag24['norm']=coag24.groupby('huc2')['accum'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
#%%
df_mrms = pd.read_feather('../output/mrms_ann_max').drop(columns=['step','heightAboveSea'])
df_mrms = df_mrms.groupby(['latitude','longitude']).median().reset_index()
df_mrms['dataset'] = 'mrms'
df_aorc = pd.read_feather('../output/aorc_ann_max')
df_aorc = df_aorc.groupby(['latitude','longitude']).median().reset_index()
df_aorc['dataset'] = 'aorc'

df_conus = pd.read_feather('../output/conus_ann_max')
df_conus = df_conus[(df_conus.season=='JJA')&(df_conus.year>=2016)].drop(columns='season')
df_conus = df_conus.groupby(['latitude','longitude']).median().reset_index()
df_conus['dataset'] = 'conus'
df = pd.concat([df_mrms,df_aorc,df_conus])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])
# %%
window = 'accum_1hr'
df1 = df.copy()
df_max = df1[['huc2',window,'dataset']].groupby(['huc2','dataset']).max().rename(columns={window:'n_max'})

df_min = df1[['huc2',window,'dataset']].groupby(['huc2','dataset']).min().rename(columns={window:'n_min'})

df1 = pd.merge(df1,df_max,on=['huc2','dataset'])
df1 = pd.merge(df1,df_min,on=['huc2','dataset'])

df1['norm_1'] = (df1[window]-df1.n_min)/(df1.n_max-df1.n_min)

window = 'accum_24hr'
df2 = df.copy()
df_max = df2[['huc2',window,'dataset']].groupby(['huc2','dataset']).max().rename(columns={window:'n_max'})

df_min = df2[['huc2',window,'dataset']].groupby(['huc2','dataset']).min().rename(columns={window:'n_min'})

df2 = pd.merge(df2,df_max,on=['huc2','dataset'])
df2 = pd.merge(df2,df_min,on=['huc2','dataset'])

df2['norm_2'] = (df2[window]-df2.n_min)/(df2.n_max-df2.n_min)


#%%
xarray_list = []
for dataset in ['conus','mrms','aorc']:
    xarray_list.append(df1[df1.dataset==dataset][['latitude','longitude','norm_1']].groupby(['latitude','longitude']).median().norm_1.to_xarray())

for dataset in ['conus','mrms','aorc']:
    xarray_list.append(df2[df2.dataset==dataset][['latitude','longitude','norm_2']].groupby(['latitude','longitude']).median().norm_2.to_xarray())


#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr

cmap = plt.get_cmap('viridis')
boundaries = np.arange(.25,1.125,.125)  # Boundaries for color changes
norm = BoundaryNorm(boundaries, cmap.N)

# Load elevation data
elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude', 'longitude']).max().elevation.to_xarray()

# Load shapefiles once
shapefiles = {}
for shape in [10, 11, 13, 14]:
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    shapefiles[shape] = gpd.read_file(shapefile_path)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15*.7, 8*.6), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    data = xarray_list[i]
    data = data.where(data > 0.25)

    # Plot elevation data
    ax.contourf(elev.longitude, elev.latitude, elev, cmap='terrain', alpha=0.7)
    
    # Plot precipitation data
    im = ax.pcolormesh(data.longitude, data.latitude, data.values, cmap=cmap, norm=norm, rasterized=True)

    # Plot shapefiles
    for gdf in shapefiles.values():
        gdf.plot(ax=ax, edgecolor='red', facecolor='none')

    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(elev.longitude.min(), elev.longitude.max())
    ax.set_ylim(elev.latitude.min(), elev.latitude.max())

    if i<3:
        scatter=ax.scatter(coag1[coag1.norm>.25].longitude,coag1[coag1.norm>.25].latitude,c=coag1[coag1.norm>.25].norm,cmap=cmap,norm=norm, edgecolor='white',s=25)

        scatter=ax.scatter(coag1[coag1.norm<=.25].longitude,coag1[coag1.norm<=.25].latitude,color='#E52B50', marker='x',s=25)
    else:
        scatter=ax.scatter(coag24[coag24.norm>.25].longitude,coag24[coag24.norm>.25].latitude,c=coag24[coag24.norm>.25].norm,cmap=cmap,norm=norm, edgecolor='white',s=25)

        scatter=ax.scatter(coag24[coag24.norm<=.25].longitude,coag24[coag24.norm<=.25].latitude,color='#E52B50', marker='x',s=25)

    #snotel = pd.read_csv('../data/snotel_loc.csv')
    #ax.scatter(snotel.Longitude,snotel.Latitude,c='white')

# Add row and column labels
row_labels = ['1-hr', '24-hr']
for row in range(2):
    fig.text(.05
    , 0.75 - row * 0.45, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=14)

col_labels = ['CONUS404', 'MRMS', 'AORC']
for col in range(3):
    fig.text(0.2 + col * 0.27, 1.0, col_labels[col], va='center', ha='center', rotation='horizontal', fontsize=12)

# Add a shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=-0.18)
cbar.set_label('Normalized Accumulation', fontsize=12)

plt.tight_layout()
#plt.subplots_adjust(top=0.9)  # Adjust top margin to fit labels
plt.show()

fig.savefig("../figures_output/2016normmap.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')