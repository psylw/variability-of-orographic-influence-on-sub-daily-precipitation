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
#%%
file_area = glob.glob('../utils/month'+'*')
area = []
for file in file_area:
    area.append(pd.read_feather(file))
area = pd.concat(area)
area = area.groupby(['latitude','longitude']).min()

area = area.to_xarray()

noband = noband.where(area.unknown>5000) # approx 0.25 timesteps

noband.to_dataframe(name='value').value.hist(bins=100)
plt.axvline(x=noband.quantile(.333), color='r', linestyle='--')  # Plot a vertical line at x=3
plt.axvline(x=noband.quantile(.667), color='r', linestyle='--')  # Plot a vertical line at x=3

q=[]
for i in range(0,9):
    q.append(noband.quantile(.125*i).values)



#%%

all_cells = pd.read_feather('../utils/above30_60') 
#all_cells2 = pd.read_feather('../utils/above30_30') 
#all_cells = pd.concat([all_cells,all_cells2 ],axis=1)
all_cells['year'] = [all_cells.start[i].year for i in all_cells.index]

#above_10 = all_cells[all_cells.area_above>=10]
above_10 = all_cells[all_cells.footprint>=10]
df = above_10[above_10.mean_rqi>=.8]
# %%


df['elev_band'] = pd.cut(df['mean_elev'], bins=q, labels=list(range(1, 9)))


#%%

df['test']=df.footprint/df.mean_time_above

q=[]
for i in range(0,5):
    q.append(df.quantile(.25*i).mean_elev)
df['elev_band'] = pd.cut(df['mean_elev'], bins=q, labels=list(range(1, 5)))

fig=plt.figure(figsize=(12, 4))

#sns.boxplot(x="elev_band", y='area_above', hue='year',data=df)
sns.boxplot(x="elev_band", y='test', hue='year',data=df)

plt.yscale('log')
plt.ylabel('min/km2')


plt.legend()
plt.show()
#fig.savefig('../fig_output/'+column+"extreme.png",
        #bbox_inches='tight',dpi=600,transparent=False,facecolor='white')

#%%
data['year_month'] = data['year'].astype(str) + '-' + data['month']
fig = plt.figure(figsize=(10, 4))
col = 'time_above40'

data = (data.groupby(['elev_band40','year_month']).count()[col]).reset_index()


sns.lineplot(data=data, x='year_month',y=col,hue='elev_band40',palette="tab10", linewidth=2.5)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.ylabel('events/month', size=12)
plt.xlabel(None)

fig.savefig('../fig_output/freq_eventshigh.png',
              bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
df['year_month'] = df['year'].astype(str) + '-' + df['month']
fig = plt.figure(figsize=(10, 4))
col = 'time_above40'

data = df[df.year>2014]
data = data[data[col]>0]

data = (data.groupby(['elev_band20','year_month']).count()[col]).reset_index()


sns.lineplot(data=data, x='year_month',y=col,hue='elev_band20',palette="tab10", linewidth=2.5)
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.ylabel('event/month')
plt.xlabel(None)

#%%
fig.savefig('../fig_output/freq_eventsall.png',
              bbox_inches='tight',dpi=600,transparent=False,facecolor='white')