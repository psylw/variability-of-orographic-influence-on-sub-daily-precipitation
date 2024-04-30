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

noband.to_dataframe(name='value').value.hist(bins=100)
plt.axvline(x=noband.quantile(.333), color='r', linestyle='--')  # Plot a vertical line at x=3
plt.axvline(x=noband.quantile(.667), color='r', linestyle='--')  # Plot a vertical line at x=3

q=[]
for i in range(0,9):
    q.append(noband.quantile(.125*i).values)



#%%
files1 = glob.glob('..\\output\\*tes')
files2 = glob.glob('..\\output\\*tal')
files3 = glob.glob('..\\output\\*elev')

all = []
for file1,file2,file3 in zip(files1,files2,files3):
       p1 = pd.read_feather(file1)
       p2 = pd.read_feather(file2)
       p3 = pd.read_feather(file3)

       p2 = p2.rename(columns={'storm_idx':'storm_id'})
       p1=pd.merge(p1, p2, on=['year', 'month', 'storm_id'], how='left')

       all.append(pd.merge(p1, p3, on=['year', 'month', 'storm_id'], how='left'))

df = pd.concat(all).reset_index().fillna(0)  
# %%


df['elev_band20'] = pd.cut(df['median_elevation20'], bins=q, labels=list(range(1, 9)))
df['elev_band30'] = pd.cut(df['median_elevation30'], bins=q, labels=list(range(1, 9)))
df['elev_band40'] = pd.cut(df['median_elevation40'], bins=q, labels=list(range(1, 9)))

#%%

df['area20_tot'] = df.area_above20/df.area_tot
df['area30_tot'] = df.area_above30/df.area_tot
df['area40_tot'] = df.area_above40/df.area_tot

columns = {'thresh20':['area_above20', 'time_above20', 'area20_tot'], 'thresh30':['area_above30', 'time_above30', 'area30_tot'],'thresh40':['area_above40', 'time_above40', 'area40_tot']}

name = {'thresh20':['area above 20 mm/hr', 'time above 20 mm/hr', 'fraction of area above 20 mm/hr'], 'thresh30':['area above 30 mm/hr', 'time above 30 mm/hr', 'fraction of area above 30 mm/hr'],'thresh40':['area above 40 mm/hr', 'time above 40 mm/hr', 'fraction of area above 40 mm/hr']}

#%%
data = df[df.year>2020]

larger = data['area_above20'].quantile(.5)
print(larger)
longer = data['time_above20'].quantile(.5)
print(longer)
data = data.loc[(data['area_above20']>larger)&(data['time_above20']>longer)]

for grp in columns.keys():
    data = data[data['area_above'+grp[-2:]]>0]
    hue = 'elev_band'+grp[-2:]

    
    for i,column in enumerate(columns[grp]):

        fig=plt.figure(figsize=(12, 4))

        sns.boxplot(x="year", y=column, hue=hue, data=data)
        if i == 0:
             plt.yscale('log')
             plt.ylabel('km2')
        elif i==1:
             plt.ylabel('min')
        else:
             plt.ylabel(None)
        plt.xlabel(None)

        plt.title(name[grp][i])

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