# map frequency of events above threshold
#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# compare median max precip at different thresholds at different windows
q = 0.5
all = []
for window in (1,3,12,24):
    print(window)
    files = glob.glob('../../output/duration_'+str(window)+'/*')

    df_all_years = []
    for file in files:

        df = pd.read_feather(file)
        df = df[(df.quant==q)]
        df_all_years.append(df)

    df_all_years = pd.concat(df_all_years)
    df_all_years['window'] = window
    all.append(df_all_years)
    ds = df_all_years.groupby(['latitude','longitude']).median().APCP_surface.to_xarray()
    ds.plot()

    plt.show()
#%%
# boxplot frequency of events above threshold, by window, hue elevation
elev = pd.read_feather('../../output/elevation')

all = pd.concat(all)
df = pd.merge(all,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

#%%
def min_max_normalize(group):
    group['normalized_max_precip'] = (group['APCP_surface'] - group['APCP_surface'].min()) / (group['APCP_surface'].max() - group['APCP_surface'].min())
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