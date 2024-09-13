#%%
# tutorial from here:
# https://nbviewer.org/github/NOAA-OWP/AORC-jupyter-notebooks/blob/master/jupyter_notebooks/AORC_Zarr_notebook.ipynb


import xarray as xr
import fsspec
import numpy as np

import dask
from dask.distributed import Client
'''import geopandas as gpd
from shapely.geometry import Point,MultiPolygon

# Load the shapefile
shapefile_path = '../data/transposition_zones_shp/Transposition_Zones.shp'
gdf = gpd.read_file(shapefile_path)

valid_polygons = []

for poly in gdf[gdf.ZONE_NAME.isin(['Front Range Transition Zone','Colorado Rockies North'])].geometry:
    if not poly.is_valid:
        poly = poly.buffer(0)  # Fix invalid polygon
    valid_polygons.append(poly)

multi_polygon = MultiPolygon(valid_polygons)'''

# %%
base_url = f's3://noaa-nws-aorc-v1-1-1km'

client = Client()
client

#%%
dataset_years = list(range(1979,2024))

#variables=['APCP_surface','UGRD_10maboveground','VGRD_10maboveground']
variables=['APCP_surface']
for year in dataset_years:
    for var in variables:

        single_year_url = f'{base_url}/{year}.zarr/'

        ds = xr.open_zarr(fsspec.get_mapper(single_year_url, anon=True), consolidated=True)

        ds=ds[var]

        co_lat_min = 36.9
        co_lon_min = 250.8-360
        co_lat_max = 41.1
        co_lon_max = 256-360
        '''        shapefile = gpd.GeoDataFrame({'geometry': multi_polygon}, crs="EPSG:4326")

        # Enable RioXarray extension
        ds = ds.rio.write_crs("EPSG:4326") 
        # Reproject the shapefile to match the dataset's CRS if needed
        shapefile = shapefile.to_crs(ds.rio.crs)
        # Clip the dataset using the shapefile
        clipped_ds = ds.rio.clip(shapefile.geometry, shapefile.crs, drop=True)

        clipped_ds = clipped_ds.where(clipped_ds.latitude>39.4,drop=True)'''


        clipped_ds = ds.sel(latitude=slice(co_lat_min,co_lat_max),longitude=slice(co_lon_min,co_lon_max))

        clipped_ds=clipped_ds.where(clipped_ds.time.dt.month.isin(range(6,9)),drop=True)
        #clipped_ds = clipped_ds.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))

        # save clip to netcdf
        clipped_ds.to_netcdf('../data/aorc/larger_aorc_'+var+'_'+str(year)+'.nc', compute=False, mode='w')

        # Trigger the actual computation with parallel processing
        with dask.config.set(scheduler='threads'):  # Use 'processes' or 'distributed' if needed
            clipped_ds.to_netcdf('../data/aorc/larger_aorc_'+var+'_'+str(year)+'.nc', mode='w')


# %%
