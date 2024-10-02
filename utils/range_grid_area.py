#%%
import pandas as pd
import xarray as xr
import numpy as np
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
name = 'nldas'
#name = 'conus'
#%%
year = 2022
########################## UNCOMMENT WHAT DATASET TO USE
#dataset = '../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
dataset = '../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
#dataset = '../data/conus404/wrf2d_d01_'+str(year)+'.nc'
##############################################################################
precip = xr.open_dataset(dataset)

########################## IF NLDAS UNCOMMENT
precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})

##############################################################################
########################## USE FOR NLDAS AND AORC
precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))
########################## IF CONUS UNCOMMENT
#precip = precip.sel(longitude = slice(-109.04,-103.96),latitude = slice(36.98,41.02))
#############################################################################
#%%
df = precip.isel(time=0).to_dataframe()
elev = pd.read_feather('../output/'+name+'_elev')
df = pd.merge(df,elev[['latitude','longitude','region']],on=['latitude','longitude'])

#%%
region_area = []
for region in range(16):
    precip = df[df.region==region].groupby(['latitude','longitude']).max().to_xarray()
    # Assuming ds is your xarray dataset with lat and lon coordinates
    lat = precip['latitude'].values
    lon = precip['longitude'].values

    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Earth radius in kilometers
    R = 6371

    # Calculate the latitudinal and longitudinal grid spacings (assumes regular grid)
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    # Create 2D arrays for latitude (cosine of latitude) and longitudinal grid
    lat_grid, lon_grid = np.meshgrid(lat_rad[:-1], lon_rad[:-1], indexing='ij')

    # Compute area of each grid cell (in km^2)
    cell_areas = (R**2) * np.abs(np.outer(dlat, dlon)) * np.cos(lat_grid)

    region_area.append({'region':region,
                  'cell_area':np.mean(cell_areas)})

df = pd.DataFrame(region_area)
df.to_feather('../output/cell_area'+name)