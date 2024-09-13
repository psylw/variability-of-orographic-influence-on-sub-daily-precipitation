# calculate area of storm above threshold for each elevation bin

#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
elev = pd.read_feather('../../output/elevation')
q = 0.5
all = []
for window in (1,3,12,24):

    files = glob.glob('../../output/duration_'+str(window)+'/*')

    for file in files:

        df = pd.read_feather(file)
        df = df[(df.quant==q)]
        df = pd.merge(df,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

        test = []
        for region in range(0,16):
            t = df[df.region==region]
            t = t.groupby(['storm_id','elevation_category']).count().latitude.reset_index()
            t['region'] = region
            t['window'] = window
            test.append(t)
        test = pd.concat(test)
        all.append(test)

# %%
df = pd.concat(all)
#%%
def min_max_normalize(group):
    group['normalized_max_precip'] = (group['latitude'] - group['latitude'].min()) / (group['latitude'].max() - group['latitude'].min())
    return group

# Apply normalization based on region and window
df = df.groupby(['region', 'window']).apply(min_max_normalize)
#%%
# Define the regions in reverse row order
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), sharex=True)
axes = axes.flatten()  # Flatten the axes array to iterate over


for idx, region in enumerate(regions):
    plot = df[df.region == region]
    sns.boxplot(data=plot, x='window', y='normalized_max_precip', hue='elevation_category', ax=axes[idx])
    axes[idx].set_title(f'Region {region}')

plt.tight_layout()
plt.show()