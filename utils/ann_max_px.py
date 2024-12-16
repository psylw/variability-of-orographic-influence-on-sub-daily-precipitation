#
# save yearly seasonal max accum for 1hr and 24hr
# resample AORC and MRMS to CONUS404 grid
# merge with elevation, HUC2, aspect, slope

# dims: year, lat, lon
#var: 1hr accum, 24hr accum, elevation, HUC2, aspect, slope

# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr
import xesmf as xe
import glob
########################## UNCOMMENT WHAT DATASET TO USE
name = 'aorc'
#name = 'mrms'
#name = 'conus'

#%%
df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016_JJA.nc')
df_conus = df_conus.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))

df_aorc = xr.open_dataset('../data/aorc/larger_aorc_APCP_surface_2016.nc')
df_aorc = df_aorc.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))

df_mrms = xr.open_dataset('../data/mrms/2022_mrms_1hr_radaronly_JJA.nc')
df_mrms = df_mrms.sel(longitude = slice(-109,-104.005),latitude = slice(41,37))

regridder_aorc = xe.Regridder(df_aorc, df_conus, "conservative")
regridder_mrms = xe.Regridder(df_mrms, df_conus, "conservative")
#%%
#files = glob.glob('../data/conus404/*.nc')
files = glob.glob('../data/mrms/*.nc')
#files = glob.glob('../data/aorc/*.nc')
ann_max = []

for file in files:

    ##############################################################################
    precip = xr.open_dataset(file)
    #precip = precip.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))
    #precip = regridder_aorc(precip)
    precip = regridder_mrms(precip)

    ########################## UNCOMMENT WHAT DATASET TO USE
    #precip = precip.rename({'APCP_surface': 'accum_1hr'})
    precip = precip.rename({'unknown': 'accum_1hr'})
    #precip = precip.rename({'ACRAINLSM': 'accum_1hr'})
    ##############################################################################
    #season = file[-6:-3] # conus only

    precip = precip.where(precip>=0)
    
    precip_24 = precip.rolling(time=24).sum()
    precip_24_max = precip_24.max(dim='time')

    precip_max = precip.max(dim='time')
    precip_max['accum_24hr']=precip_24_max.accum_1hr

    precip_max['year'] = precip.time.dt.year.max().values
    #precip_max['season'] = season
    print(file)
    ann_max.append(precip_max.to_dataframe().reset_index())
#%%
ann_max = pd.concat(ann_max)

ann_max.to_feather('../output/'+name+'_ann_max')


# %%
