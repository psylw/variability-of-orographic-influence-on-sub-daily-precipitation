# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr
import xesmf as xe


#%%
for window in (1,5,15,1*30,3*30,12*30,24*30):
#for window in (1*30,3*30,12*30,24*30):
    ann_max = []
    print(window)
    for idx,year in enumerate(range(2016,2023)):

        print(year)

        dataset = ['../../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_jun_rate_CO.grib2','../../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_jul_rate_CO.grib2','../../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_aug_rate_CO.grib2']
        #dataset = '../../output/mrms_nldas/mrms_nldasgrid_'+str(year)+'.nc'
        precip = xr.open_mfdataset(dataset, chunks={'time': '1GB'})

        precip['longitude'] = precip['longitude']-360

        precip = precip.rename({'unknown': 'accum'})

        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

        precip = precip.where(precip>=0)*(2/60)

        precip = precip.rolling(time=window).sum()
        precip_max = precip.max(dim='time')

        ann_max.append(precip_max)

    ann_max = xr.concat(ann_max, dim='year')

    #ann_max.to_netcdf('../../output/mrms_nldas_ann_max_px_2016_window_'+str(int(window*2))+'.nc')
    ann_max.to_netcdf('../../output/mrms_ann_max_px_window_'+str(int(window*2))+'.nc')


# %%
