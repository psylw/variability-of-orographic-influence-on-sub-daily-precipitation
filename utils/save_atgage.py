# save nonzero values at coagmet gage locations
# 1hr and 24hr accumulations
# CONUS404, MRMS, AORC, PRISM

#%%
import pandas as pd
import xarray as xr
import glob
import numpy as np
import xesmf as xe
name = 'mrms'
#name = 'conus'
#name = 'aorc'
#%%
# open all gage data
coag = pd.read_feather('../output/coag_1')
#merge with gage data
coag_coords = coag.groupby(['latitude','longitude']).max().reset_index()[['latitude','longitude']]

#%%
#files = glob.glob('../data/conus404/*JJA*.nc')[-7::]
#files = glob.glob('../data/mrms/*.nc')
files = glob.glob('../data/aorc/*.nc')
ann_max = []

for file in files:

    ##############################################################################
    precip = xr.open_dataset(file)
    precip = precip.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))
    #precip = precip.sel(longitude = slice(-109,-104.005),latitude = slice(41,37))

    ########################## UNCOMMENT WHAT DATASET TO USE
    precip = precip.rename({'APCP_surface': 'accum_1hr'})
    #precip = precip.rename({'unknown': 'accum_1hr'})
    #precip = precip.rename({'ACRAINLSM': 'accum_1hr'})
    ##############################################################################
    for coord in coag_coords.index:
        data = {'time':precip.time.values,
'latitude':coag_coords.latitude[coord],
'longitude':coag_coords.longitude[coord],
'accum_1hr':precip.sel(latitude = coag_coords.latitude[coord],longitude= coag_coords.longitude[coord],method='nearest').accum_1hr.values,
'accum_24hr':precip.sel(latitude = coag_coords.latitude[coord],longitude= coag_coords.longitude[coord],method='nearest').rolling(time=24).sum().accum_1hr.values}

        sample = pd.DataFrame(data=data)
        

        ann_max.append(sample)
    print(file)
# %%
ann_max = pd.concat(ann_max)

ann_max.to_feather('../output/'+name+'_atgage')
# %%
