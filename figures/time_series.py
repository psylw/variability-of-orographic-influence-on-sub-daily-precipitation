# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde
#%%
coag1 = pd.read_feather('../output/coag_zero')

coag_coord = coag1.groupby(['latitude','longitude']).max().reset_index()

elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude', 'longitude']).max().to_xarray()

huc = []
for coord in coag_coord.index:
    data = {'latitude':coag_coord.latitude[coord],
    'longitude':coag_coord.longitude[coord],
    'huc2':elev.sel(latitude = coag_coord.latitude[coord],longitude= coag_coord.longitude[coord],method='nearest').huc2.values,
    'elevation':elev.sel(latitude = coag_coord.latitude[coord],longitude= coag_coord.longitude[coord],method='nearest').elevation.values}
    huc.append(data)

huc = pd.DataFrame(huc)

coag1 = pd.merge(coag1,huc,on=['latitude','longitude'])


#%%
aorc1 = pd.read_feather('../output/aorc_atgage')

conus1 = pd.read_feather('../output/conus_new_atgage')

mrms1 = pd.read_feather('../output/mrms_atgage')

aorc1['year'] = aorc1.time.dt.year
conus1['year'] = conus1.time.dt.year
mrms1['year'] = mrms1.time.dt.year
aorc1 = pd.merge(aorc1,huc,on=['latitude','longitude'])
conus1  = pd.merge(conus1,huc,on=['latitude','longitude'])
mrms1 = pd.merge(mrms1,huc,on=['latitude','longitude'])

coag1['ds'] = 'coag'
aorc1['ds'] = 'aorc'
conus1['ds'] = 'conus'
mrms1['ds'] = 'mrms'
df = pd.concat([coag1.drop(columns=['id','index']),aorc1,conus1,mrms1])
df['huc2'] = df.huc2.astype('float')

#df = df[df.elevation>2000]
#%%
plot = df.groupby(['latitude','longitude','year','ds']).max().reset_index()

sns.kdeplot(data =plot,x='accum_1hr',hue='ds')
#sns.boxplot(data =plot, x='huc2',y='accum_24hr',hue='ds')
sns.histplot(data =plot,x='accum_1hr',hue='ds',multiple="stack",)
    
#%%
sns.boxplot(data =plot, x='huc2',y='accum_24hr',hue='ds')
plt.ylim(0,60)

#%%
window = 12
index = window*2+1

def create_0_24hr_index(g):
    """
    g is a subset of df for one (id, year).

    Steps:
      1) Find the time of the maximum 'accum' (first occurrence if ties).
      2) Construct 25 integer offsets: 0..24.
         - Conceptually, offset 12 is the "peak" hour,
           offset 0 is 12 hrs before, offset 24 is 12 hrs after.
      3) Optionally store the original time each offset corresponds to (if it exists).
      4) Return a new DataFrame with 'hour_offset' = 0..24 as the index (or a column).
    """

    # 1) Time of max 'accum'
    idx_max = g['accum_24hr'].idxmax()
    max_time = g.loc[idx_max, 'time']

    # 2) Create hour_offset range 0..24 (25 steps)
    hour_offsets = range(index)  # 0,1,2,...,24

    # 3) Compute the actual timestamp for each offset:
    #    - offset=0  => max_time - 12h
    #    - offset=12 => max_time (peak)
    #    - offset=24 => max_time + 12h
    start_time = max_time - pd.Timedelta(hours=index-1)
    offset_times = [start_time + pd.Timedelta(hours=h) for h in hour_offsets]

    # We'll build a small reference DF that has (hour_offset, time)
    offset_df = pd.DataFrame({
        'hour_offset': hour_offsets,
        'target_time': offset_times  # keep as reference if you like
    })

    # Now, if you want to know which *original* row aligns with each offset_time,
    # you can merge back to 'g' on the timestamp. Let's do a left-merge so that
    # we keep all hour_offsets even if the original time didn't exist.
    g_indexed = g.set_index('time')  # set time as index for merging
    merged = pd.merge(offset_df, g_indexed, left_on='target_time', right_index=True, how='left')

    # 'merged' now has 25 rows for each group, with hour_offset 0..24,
    # plus any columns from 'g' (accum, latitude, etc.) if they match the exact timestamp.

    # Optionally fill missing accum, etc., if the time didn't exist
    # merged['accum'] = merged['accum'].fillna(0)

    # Return a neat DataFrame with hour_offset as *the* index (or a column).
    merged = merged.set_index('hour_offset')
    
    return merged

#%%

for h in [10,11,13,14]:
    for d in df.ds.unique():
        sample = df[df.ds==d]

        df_12hr = (
        sample.groupby(['latitude','longitude', 'year'], group_keys=False)  # group_keys=False to avoid extra index levels
        .apply(create_0_24hr_index)
        .reset_index())
        
        sns.lineplot(data=df_12hr[df_12hr.huc2==h],x='hour_offset',y='accum_1hr',label=d)
    plt.legend()
    plt.title(h)
    plt.show()

#%%
