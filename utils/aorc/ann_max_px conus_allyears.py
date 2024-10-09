# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr
import xesmf as xe

name = 'conus'

#%%
for window in (1,3,12,24):
    ann_max = []

    for idx,year in enumerate(range(1980,2023)):
        print(year)
        
        dataset = '../../data/conus404/wrf2d_d01_'+str(year)+'.nc'
        
        precip = xr.open_dataset(dataset)

        precip = precip.rename({'ACRAINLSM': 'accum'})
        
        precip = precip.sel(longitude = slice(-109.04,-103.96),latitude = slice(36.98,41.02))

        precip = precip.where(precip>=0)

        precip = precip.rolling(time=window).sum()*(1/window)
        precip_max = precip.max(dim='time')

        ann_max.append(precip_max)

    ann_max = xr.concat(ann_max, dim='year')
    #ann_max = ann_max.quantile(.9,dim='year').accum
    #ann_max = ann_max.quantile(.5,dim='year').accum

    ann_max.to_netcdf('../../output/'+name+'_ann_max_px_allyears'+'_window_'+str(window)+'.nc')


# %%
