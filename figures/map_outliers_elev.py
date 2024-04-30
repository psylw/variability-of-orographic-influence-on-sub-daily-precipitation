#%%
import glob
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import xarray as xr
import matplotlib.gridspec as gridspec
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature
import cartopy.crs as ccrs

#%%
#%%
files1 = glob.glob('..\\output\\*tes')[-15:]
files2 = glob.glob('..\\output\\*tal')[-15:]
files3 = glob.glob('..\\output\\*elev')[-15:]
files4 = glob.glob('..\\output\\*roid')
#%%
all = []
for file1,file2,file3,file4 in zip(files1,files2,files3,files4):
       p1 = pd.read_feather(file1)
       p2 = pd.read_feather(file2)
       p3 = pd.read_feather(file3)
       p4 = pd.read_feather(file4)

       p2 = p2.rename(columns={'storm_idx':'storm_id'})
       p1=pd.merge(p1, p2, on=['year', 'month', 'storm_id'], how='left')
       p1=pd.merge(p1, p3, on=['year', 'month', 'storm_id'], how='left')
       all.append(pd.merge(p1, p4, on=['year', 'month', 'storm_id'], how='left'))

df = pd.concat(all).reset_index().fillna(0) 
df = df[df.median_elevation40>0]
#%%
larger = df['area_above40'].quantile(.5)
print(larger)
longer = df['time_above40'].quantile(.5)
print(longer)
df = df.loc[(df['area_above40']>larger)&(df['time_above40']>longer)]
#%%
# open mrms
data_folder = os.path.join('..', '..','..','data','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')
month = xr.open_dataset(filenames[0], engine = "cfgrib",chunks={'time': '500MB'})

month = month.where(month.longitude<=256,drop=True)

datafile1 = "../../../data/elev_data/CO_SRTM1arcsec__merge.tif"
codtm =xr.open_rasterio(datafile1)
newelev = codtm.drop_vars('band')
noband = newelev.sel(band=0)
noband['x'] = noband.x+360
noband = noband.sel(y=month.latitude,x=month.longitude,method='nearest',drop=True)

q1 = noband.quantile(.333)
q2 = noband.quantile(.667)

#%%

gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
fig,axs = plt.subplots(1,1, subplot_kw=dict(projection=plotcrs), figsize=(12,8))

lon, lat = np.meshgrid(noband.x,noband.y)
levels = [1500,q1.values,q2.values,4500]
elev=plt.contourf(lon,lat, noband, levels=levels, origin='upper',cmap='terrain', 
            alpha=0.4,transform=ccrs.PlateCarree(),extend='both')

y1 = df[df.year==2021]
y2 = df[df.year==2022]
y3 = df[df.year==2023]

#plt.scatter(y1.centroid40_lon,y1.centroid40_lat,transform=ccrs.PlateCarree(),color='red',marker='x')
#plt.scatter(y2.centroid40_lon,y2.centroid40_lat,transform=ccrs.PlateCarree(),color='blue',marker='x')
plt.scatter(y3.centroid40_lon,y3.centroid40_lat,transform=ccrs.PlateCarree(),color='black',marker='x')
fig.tight_layout()
fig.subplots_adjust(right=.85)
cbar_ax = fig.add_axes([.9, 0.15, 0.03, 0.7])
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
cb=fig.colorbar(elev, cax=cbar_ax)

cb.ax.tick_params(labelsize=12)
cb.set_label("elevation (m)", fontsize=12)

axs.add_feature(cfeature.STATES, linewidth=1)


axs.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')


gl = axs.gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0, 'fontsize':12}
gl.ylabel_style = {'fontsize':12}
# add these before plotting
gl.bottom_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

gl = axs.gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0, 'fontsize':12}
gl.ylabel_style = {'fontsize':12}

#%%
fig.savefig("../fig_output/map_bands.png",
           bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
