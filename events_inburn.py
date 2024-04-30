#%%
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import rioxarray

df = pd.read_feather('output/count_duration_threshold_px2')
#df['year'] = [df.time[i].year for i in df.index]
# Replace 'file_path.geojson' with the path to your GeoJSON file
file_path = 'WFIGS_Interagency_Perimeters_1406150765430228857.geojson'

# Read the GeoJSON file using geopandas
gdf = gpd.read_file(file_path)
# %%
gdf = gdf[gdf.poly_IncidentName.isin(['East Troublesome','Cameron Peak'])]

shapefile_path = "transposition_zones_shp/Transposition_Zones.shp"
gdf2 = gpd.read_file(shapefile_path)

test2 = df.groupby(['time','latitude','longitude']).max().to_xarray()
crs = gdf.crs
test2.rio.write_crs(crs, inplace=True)
test2['longitude'] = test2['longitude']-360

clipped_cpf = test2.rio.clip(gdf.geometry.values[0], crs=gdf.crs)
clipped_et = test2.rio.clip(gdf.geometry.values[1], crs=gdf.crs)
clipped_zone5 = test2.rio.clip([gdf2[gdf2.TRANS_ZONE==5].geometry.values[0]], crs=gdf.crs)

#%%
precip_cpf = clipped_et.to_dataframe().reset_index().dropna()



precip_cpf['month'] = [precip_cpf.time[i].month for i in precip_cpf.index]
precip_cpf['year'] = [precip_cpf.time[i].year for i in precip_cpf.index]

precip_cpf['year_month'] = precip_cpf['year'].astype(str) + '-' + precip_cpf['month'].astype(str)
fig = plt.figure(figsize=(12, 4))


data = precip_cpf[precip_cpf.unknown>=60]
data = data.groupby(['year_month','time']).count()

sns.lineplot(data=data, x='year_month',y='unknown',palette="tab10", linewidth=2.5)
plt.xticks(rotation=45)
plt.legend(loc='center left', bbox_to_anchor=(1,.5))
plt.ylabel('event/month-km2')
plt.xlabel(None)
#plt.title('frequency of events per month, normalized by transposition area')

#%%
