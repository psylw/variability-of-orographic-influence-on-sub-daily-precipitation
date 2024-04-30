###############################################################################
# plot storm attributes by transposition zone
###############################################################################
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


files1 = glob.glob('..\\output\\*tes')
files2 = glob.glob('..\\output\\*tal')

all = []
for file1,file2 in zip(files1,files2):
       p1 = pd.read_feather(file1)
       p2 = pd.read_feather(file2)

       p2 = p2.rename(columns={'storm_idx':'storm_id'})

       all.append(pd.merge(p1, p2, on=['year', 'month', 'storm_id'], how='left'))
#%%
df = pd.concat(all).reset_index().fillna(0)        
shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)

df['zone_name'] = [gdf.loc[gdf.TRANS_ZONE==df.max_intersecting_geometry_id[i]].ZONE_NAME.values[0] for i in range(len(df))]

df['area20_tot'] = df.area_above20/df.area_tot
df['area30_tot'] = df.area_above30/df.area_tot
df['area40_tot'] = df.area_above40/df.area_tot

columns = ['area_above20', 'area_above30', 'area_above40', 'area_tot',
       'time_above20', 'time_above30', 'time_above40', 'time_tot','area20_tot','area30_tot','area40_tot']

name = [ 'area above 20 mm/hr', 'area above 30 mm/hr', 'area above 40 mm/hr','total area', 
       'time above 20 mm/hr', 'time above 30 mm/hr', 'time above 40 mm/hr', 'total time','area20_tot','area30_tot','fraction of area above 40 mm/hr']
#%%
columns = ['area_above40','time_above40', 'area40_tot']

name = [ 'area above 40 mm/hr','time above 40 mm/hr', 'fraction of area above 40 mm/hr']
#%%
# select outliers
data = df[df['area_above40'] > 0]
outliers_zone = df.groupby('zone_name').quantile(.9).reset_index()

new_df = []
for zone in df.zone_name.unique():
       z = data[data.zone_name==zone]
       outlier = outliers_zone[outliers_zone.zone_name==zone]
       for column in columns:
              d = pd.DataFrame(z[z[column]>outlier[column].values[0]][[column,'year']])
              d['zone_id'] = zone
              new_df.append(d)
new_df = pd.concat(new_df)
#%%
# area and duration distributions
#data = df[df.year>2020]
data = df.loc[df.max_intersecting_geometry_id.isin([1,3,5])]
#data = data.loc[(data['area_above20']>data['area_above20'].quantile(.5))&(data['time_above20']>data['time_above20'].quantile(.5))]

for i,column in enumerate(columns):
       fig=plt.figure(figsize=(12, 4))
       data = data[data[column]>0]
       if i == 0:
              plt.yscale('log')

       sns.boxplot(x="year", y=column, hue='zone_name', data=data)
       '''       if i < 4:
              plt.yscale('log')
              plt.ylabel('km2')

       elif i<10:
              plt.ylabel('min')
       else:
              plt.ylabel(None)'''
      # plt.yscale('log')
       plt.xlabel(None)
       plt.title(name[i])
       #plt.ylim(0,ylim)
       plt.legend(loc='lower center', bbox_to_anchor=(.5,-1))
       plt.show()
       fig.savefig('../fig_output/'+column+".png",
              bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
#%%
# area vs duration
from sklearn.metrics import r2_score
fit = []

data = df[df['area_above40'] > 0]
for zone in df.zone_name.unique():
       for year in df.year.unique():
              plot = data.loc[(data.year == year)&(data.zone_name == zone)]
              # Perform linear regression
              x = np.log(plot.area_above40)
              y = plot.time_above40
              # Perform linear regression to fit a line to the data
              coefficients = np.polyfit(x, y, 1)
              poly = np.poly1d(coefficients)
              y_pred = poly(x)
              r_squared = r2_score(y, y_pred)
              fit.append([zone,year,r_squared,coefficients[0],coefficients[1]])
fit = pd.DataFrame(fit,columns=['zone','year','r_squared','slope','intercept'])
#%%
# calculate transposition zone area
# open mrms
data_folder = os.path.join('..', '..','..','data','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')
month = xr.open_dataset(filenames[0], engine = "cfgrib",chunks={'time': '500MB'})

month = month.where(month.longitude<=256,drop=True)
# open transposition zone as tif
ds = xr.open_rasterio('..//transposition_zones_shp//trans_zones.tif')
ds['x'] = ds.x+360
ds = ds.sel(y=month.latitude,x=month.longitude,method='nearest',drop=True)

area_zone = []
for i in df.max_intersecting_geometry_id.unique():
       area = len(ds.where(ds==i).to_dataframe(name='value').dropna())
       area_zone.append([i,area])

area_zone = pd.DataFrame(area_zone,columns=['zone_id','zone_area'])
df['area_zone']=[area_zone.loc[area_zone.zone_id==df.max_intersecting_geometry_id[i]].zone_area.values[0] for i in range(len(df))]
#%%
df['year_month'] = df['year'].astype(str) + '-' + df['month']
fig = plt.figure(figsize=(12, 4))
col = 'time_above40'
#data = df.loc[df.max_intersecting_geometry_id.isin([1,3,5])]
data = df[df[col]>0]

#data = data[data['year'] > 2020]
data = (data.groupby(['zone_name','year_month','area_zone']).count()[col]).reset_index()
data['time_above40'] = data['time_above40']/data['area_zone']
# normalize with area

sns.lineplot(data=data, x='year_month',y='time_above40',hue='zone_name',palette="tab10", linewidth=2.5)
plt.xticks(rotation=45)
plt.legend(loc='center left', bbox_to_anchor=(1,.5))
plt.ylabel('event/month-km2')
plt.xlabel(None)
plt.title('frequency of events per month, normalized by transposition area')
fig.savefig('../fig_output/freq_events.png',
              bbox_inches='tight',dpi=600,transparent=False,facecolor='white')



# %%
