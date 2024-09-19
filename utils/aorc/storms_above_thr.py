# for each threshold, based on max annual intensities for each region, find independent storms for each year
#%%
import pandas as pd
import xarray as xr
########################## UNCOMMENT WHAT DATASET TO USE
#name = 'aorc'
name = 'nldas'
#name = 'conus'
#%%
for window in (1,3,12,24):

    annual_max = pd.read_feather('../../output/'+name+'_ann_max_region'+'_window_'+str(window))

    thr = annual_max.groupby(['latitude','longitude']).median().min(axis=1).to_xarray()

    storm = []

    for idx,year in enumerate(range(2002,2023)):
        print(year)
        ########################## UNCOMMENT WHAT DATASET TO USE
        #dataset = '../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc'
        dataset = '../../data/NLDAS/NLDAS_FORA0125_H.A'+str(year)+'.nc'
        #dataset = '../../data/conus404/wrf2d_d01_'+str(year)+'.nc'
        ##############################################################################
        precip = xr.open_dataset(dataset)
        ########################## IF NLDAS UNCOMMENT
        precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'})
        ##############################################################################
        ########################## UNCOMMENT WHAT DATASET TO USE
        #precip = precip.rename({'APCP_surface': 'accum'})
        precip = precip.rename({'Rainf': 'accum'})
        #precip = precip.rename({'ACRAINLSM': 'accum'})
        ##############################################################################
        ########################## USE FOR NLDAS AND AORC
        precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))
        ########################## IF CONUS UNCOMMENT
        #precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37.1,40.9))
        ##############################################################################

        precip = precip.where(precip>=0)

        size_sub_lat = int(len(precip.latitude)/4)
        size_sub_lon = int(len(precip.longitude)/4)

        ds_daily = precip.rolling(time=window).sum()
        ds_aggregated = ds_daily.coarsen(latitude=size_sub_lat, longitude=size_sub_lon).max()

        ds_aggregated = ds_aggregated.fillna(0)

        # Create a mask where the variable is zero
        zero_mask = ds_aggregated['accum'] == 0

        # Rolling sum over 12 hours (axis 0 corresponds to the time dimension)
        rolling_sum = zero_mask.rolling(time=12, center=False).sum()

        # Identify where the sum equals 12 (indicating 12 consecutive hours of zero values)
        segments = (rolling_sum >= 12)

        # Generate unique IDs by cumulative sum where segments start
        id_var = segments.cumsum(dim='time')

        # Assign the ID variable to the dataset
        ds_aggregated['id_var'] = id_var
        ds_aggregated['id_var'] = ds_aggregated['id_var'].where(~segments)

        precip_above_ann_max = ds_aggregated.where(ds_aggregated.accum>=thr,drop=True)
        ########################## IF CONUS UNCOMMENT
        #storm.append(precip_above_ann_max.to_dataframe().drop(columns=['XLAT','XLONG']).dropna().reset_index())
        ##############################################################################
        storm.append(precip_above_ann_max.to_dataframe().dropna().reset_index())
        
    pd.concat(storm).reset_index().to_feather('../../output/'+name+'_storms_above_thr'+'_window_'+str(window))

# %%
