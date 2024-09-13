# get the annual max intensity for each region at various window intervals
#%%
import pandas as pd
import xarray as xr

for window in (1,3,12,24):
    ann_max = []
    for idx,year in enumerate(range(1979,2024)):
        print(year)

        precip = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc')

        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

        precip = precip.where(precip>=0)

        size_sub_lat = int(len(precip.latitude)/4)
        size_sub_lon = int(len(precip.longitude)/4)

        ds_daily = precip.rolling(time=window).sum()
        ds_aggregated = ds_daily.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()

        ann_max.append(ds_aggregated.max(dim='time').to_dataframe().rename(columns={'APCP_surface':'year'+str(year)}))

    ann_max = pd.concat(ann_max,axis=1)
    ann_max.reset_index().to_feather('../../output/ann_max_region'+'_window_'+str(window))
# %%
