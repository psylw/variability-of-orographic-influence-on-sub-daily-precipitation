# get the duration above annual max intensity 
#%%
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import rasterio
from datetime import datetime, timedelta


# Function to calculate burstiness with consideration of inter-event time equal to 1 being the same event
def burstiness(time_series, threshold):
    # Identify event times where precipitation exceeds the threshold
    event_times = np.where(time_series > threshold)[0]
    
    # If there are fewer than 2 events, return NaN (cannot calculate burstiness)
    if len(event_times) < 2:
        return np.nan
    
    # Aggregate consecutive events with inter-event times of 1
    distinct_event_times = [event_times[0]]  # Start with the first event time
    for i in range(1, len(event_times)):
        if event_times[i] - event_times[i - 1] > 1:
            distinct_event_times.append(event_times[i])
    
    # If there are still fewer than 2 distinct events, return NaN
    if len(distinct_event_times) < 2:
        return np.nan
    
    # Calculate inter-event times between distinct events
    inter_event_times = np.diff(distinct_event_times)
    
    # Calculate mean and standard deviation of inter-event times
    mean_tau = np.mean(inter_event_times)
    std_tau = np.std(inter_event_times)
    
    # Calculate burstiness
    if mean_tau + std_tau == 0:
        return 0  # Avoid division by zero
    burstiness = (std_tau - mean_tau) / (std_tau + mean_tau)
    
    return burstiness

# %%
window = 24
storms = pd.read_feather('../../output/storms_above_thr'+'_window_'+str(window))
storms = storms.drop(columns='index')
storms['year'] = storms.time.dt.year

annual_max = pd.read_feather('../../output/ann_max_region'+'_window_'+str(window))

for year in range(1979,2024):
    print(year)

    precip = xr.open_dataset('../../data/aorc/larger_aorc_APCP_surface_'+str(year)+'.nc')

    precip = precip.sel(longitude = slice(-109,-104),latitude = slice(37,41))

    precip = precip.where(precip>=0)
    
    ds_daily = precip.rolling(time=window).sum()

    size_sub_lat = int(len(precip.latitude)/4)
    size_sub_lon = int(len(precip.longitude)/4)

    s = storms[storms.year==year]

    # create dictionary for coordinates to reference original coordinates
    expand_lat = {}

    for i,lat in enumerate(np.sort(storms.latitude.unique())):
        expand_lat[lat] = precip.latitude.values.reshape(int(len(precip.latitude)/size_sub_lat), size_sub_lat)[i]

    expand_lon = {}

    for i,lon in enumerate(np.sort(storms.longitude.unique())):
        expand_lon[lon] = precip.longitude.values.reshape(int(len(precip.longitude)/size_sub_lon), size_sub_lon)[i]

    s = s.groupby(['latitude','longitude','id_var']).agg(list).reset_index()
    region = s.groupby(['latitude','longitude']).max().reset_index()[['latitude','longitude']]
    region['region'] = region.index
    s = pd.merge(s, region, on=['latitude', 'longitude'], how='left')

    burst = []

    quantile = np.arange(0,1,.125)
    
    for i in s.index:
        time_idx = np.argmax(s.iloc[i].APCP_surface)
        sample = precip.sel(time=slice(s.iloc[i].time[time_idx]-timedelta(hours=window),s.iloc[i].time[time_idx]+timedelta(hours=window)),
                                longitude = slice(np.min(expand_lon[s.iloc[i].longitude]),np.max(expand_lon[s.iloc[i].longitude])),latitude = slice(np.min(expand_lat[s.iloc[i].latitude]),np.max(expand_lat[s.iloc[i].latitude])))

        for q in quantile:
            threshold = annual_max[(annual_max.latitude==s.iloc[i].latitude)&(annual_max.longitude==s.iloc[i].longitude)].groupby(['latitude','longitude']).median().quantile(q,axis=1).values[0]/window

            burstiness_per_coordinate = xr.apply_ufunc(
                burstiness,
                sample,
                kwargs={'threshold': threshold},
                input_core_dims=[['time']],  # Apply over the 'time' dimension
                vectorize=True  # Apply function element-wise over lat/lon
            )
            b = burstiness_per_coordinate.to_dataframe().dropna()
            if not b.empty:
                b['quant'] = q
                b['year'] = year
                b['region'] = s.iloc[i].region
                b['storm_id'] = s.iloc[i].id_var
                b['threshold'] = threshold
                burst.append(b)
                

# %%
