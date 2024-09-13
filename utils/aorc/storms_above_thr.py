# for each threshold, based on max annual intensities for each region, find independent storms for each year
#%%
import pandas as pd
import xarray as xr


for window in (1,3,12,24):
    annual_max = pd.read_feather('../../output/ann_max_region'+'_window_'+str(window))
    thr = annual_max.groupby(['latitude','longitude']).median().min(axis=1).to_xarray()

    storm = []
    for year in range(1979,2024):
        print(year)

        precip = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc')
        #precip = precip.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

        precip = precip.where(precip>=0)

        size_sub_lat = int(len(precip.latitude)/4)
        size_sub_lon = int(len(precip.longitude)/4)

        ds_daily = precip.rolling(time=window).sum()
        ds_aggregated = ds_daily.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()
        ds_aggregated = ds_aggregated.fillna(0)

        # Create a mask where the variable is zero
        zero_mask = ds_aggregated['APCP_surface'] == 0

        # Rolling sum over 12 hours (axis 0 corresponds to the time dimension)
        rolling_sum = zero_mask.rolling(time=12, center=False).sum()

        # Identify where the sum equals 12 (indicating 12 consecutive hours of zero values)
        segments = (rolling_sum >= 12)

        # Generate unique IDs by cumulative sum where segments start
        id_var = segments.cumsum(dim='time')

        # Assign the ID variable to the dataset
        ds_aggregated['id_var'] = id_var
        ds_aggregated['id_var'] = ds_aggregated['id_var'].where(~segments)

        precip_above_ann_max = ds_aggregated.where(ds_aggregated.APCP_surface>=thr,drop=True)

        storm.append(precip_above_ann_max.to_dataframe().dropna().reset_index())
        
    pd.concat(storm).reset_index().to_feather('../../output/storms_above_thr'+'_window_'+str(window))
# %%
