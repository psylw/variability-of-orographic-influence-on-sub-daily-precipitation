# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr
import xesmf as xe
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
name = 'nldas'
#name = 'conus'

#%%
df_nldas = xr.open_dataset('../../data/NLDAS/NLDAS_FORA0125_H.A2016.nc')
df_nldas = df_nldas.rename({'lat': 'latitude', 'lon': 'longitude'})
df_aorc = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_2016.nc')
df_conus = xr.open_dataset('../../data/conus404/wrf2d_d01_2016.nc')

regridder_aorc = xe.Regridder(df_aorc, df_nldas, "bilinear")
regridder_conus = xe.Regridder(df_conus, df_nldas, "bilinear")
#%%
for window in (1,3,12,24):

    ann_max = []

    for idx,year in enumerate(range(2016,2023)):

        print(year)
        ########################## UNCOMMENT WHAT DATASET TO USE
        #dataset = '../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
        dataset = '../../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
        #dataset = '../../data/conus404/wrf2d_d01_'+str(year)+'.nc'

        ##############################################################################
        precip = xr.open_dataset(dataset)

        #precip = regridder_aorc(precip)
        #precip = regridder_conus(precip)

        ########################## IF NLDAS UNCOMMENT
        precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
        ##############################################################################
        ########################## UNCOMMENT WHAT DATASET TO USE
        #precip = precip.rename({'APCP_surface': 'accum'})
        precip = precip.rename({'Rainf': 'accum'})
        #precip = precip.rename({'ACRAINLSM': 'accum'})
        ##############################################################################

        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

        precip = precip.where(precip>=0)
        
        precip = precip.rolling(time=window).sum()*(1/window)
        precip_max = precip.max(dim='time')

        ann_max.append(precip_max)

    ann_max = xr.concat(ann_max, dim='year')

    ann_max.to_netcdf('../../output/'+name+'_ann_max_px_2016'+'_window_'+str(window)+'.nc')

# %%
