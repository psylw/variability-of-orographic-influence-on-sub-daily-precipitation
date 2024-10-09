# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr
import xesmf as xe

#%%
df_mrms = xr.open_dataset('../../../data/MRMS/2min_rate_cat_month_CO/2015_aug_rate_CO.grib2')
df_nldas = xr.open_dataset('../../data/NLDAS/NLDAS_FORA0125_H.A2016.nc')
df_nldas = df_nldas.rename({'lat': 'latitude', 'lon': 'longitude'})

regridder = xe.Regridder(df_mrms, df_nldas, "bilinear")

#%%

for idx,year in enumerate(range(2016,2023)):

    print(year)

    dataset = ['../../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_jun_rate_CO.grib2','../../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_jul_rate_CO.grib2','../../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_aug_rate_CO.grib2']

    precip = xr.open_mfdataset(dataset, chunks={'time': '500MB'})
    precip = regridder(precip)

    precip = precip.rename({'unknown': 'accum'})

    precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

    precip = precip.where(precip>=0)


    precip.to_netcdf('../../output/mrms_nldas/mrms_nldasgrid_'+str(year)+'.nc')


# %%
