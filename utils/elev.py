#%%
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import rasterio

#%%

name = 'conus'

precip = pd.read_feather('../output/conus_ann_max')[['latitude','longitude','accum_1hr']].groupby(['latitude','longitude']).max().to_xarray()
#%%

# Load the source raster (the one to be resampled)
source_raster = rioxarray.open_rasterio("../../../data/elev_data/CO_SRTM1arcsec__merge.tif")

target_raster = precip.rio.write_crs("EPSG:4326")
##############################################################################

# Resample the source raster to the target raster's grid using bilinear interpolation
resampled_raster = source_raster.rio.reproject_match(target_raster, resampling=rasterio.enums.Resampling.bilinear)

resampled_raster=resampled_raster.isel(band=0).where(resampled_raster>0).to_dataframe(name='elevation').drop(columns=['spatial_ref']).reset_index().rename(columns={'y':'latitude','x':'longitude'})
#%%
import geopandas as gpd

shp = []
for shape in [10,11,13,14]:
# Load the shapefile
    shapefile_path = "../data/huc2/WBD_"+str(shape)+"_HU2_Shape/WBDHU2.shp"  # Replace with the correct path
    gdf = gpd.read_file(shapefile_path)
    shp.append(gdf)


# Combine all GeoDataFrames
combined_gdf = gpd.GeoDataFrame(pd.concat(shp, ignore_index=True))[['huc2','geometry','name']]

combined_gdf = combined_gdf.to_crs({'init': 'epsg:4326'})
#%%
from shapely.geometry import Point
coord = resampled_raster[['latitude','longitude']].groupby(['latitude','longitude']).max().reset_index()
geometry = [Point(xy) for xy in zip(coord['longitude'], coord['latitude'])]
gdf_points = gpd.GeoDataFrame(coord, geometry=geometry, crs="EPSG:4326")

gdf_points = gpd.sjoin(gdf_points, combined_gdf, how="left", predicate="within")
# %%
gdf_points["huc2"] = gdf_points["huc2"].astype(float)
#%%
df = pd.merge(resampled_raster,gdf_points[['latitude','longitude','huc2']],on=['latitude','longitude'])
#%%

def categorize_elevation(group):
    group['e_cat'] = pd.qcut(group['elevation'], q=4, labels=['q1','q2','q3','q4'])
    return group

# Apply categorization within each group
df = df.groupby('huc2').apply(categorize_elevation)
# %%
df=df.drop(columns='band')
df.to_feather('../output/'+name+'_elev')

