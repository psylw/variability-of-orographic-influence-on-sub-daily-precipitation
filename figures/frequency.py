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
    all_files = glob.glob('../output/duration_'+str(window)+'/*')
    files_nldas = glob.glob('../output/duration_'+str(window)+'/*nldas'+'*')
    aorc_files = [item for item in all_files if item not in files_nldas]

    df_all_years = []
    #for file in aorc_files:
    for file in files_nldas:

        df = pd.read_feather(file)
        df = df[(df.quant==q)]
        df_all_years.append(df)

    df_all_years = pd.concat(df_all_years)
    df_all_years['window'] = window
    all.append(df_all_years)
    ds = df_all_years.groupby(['latitude','longitude']).count().max_precip.to_xarray()
    ds.plot()

    plt.show()
#%%
# boxplot frequency of events above threshold, by window, hue elevation
elev = pd.read_feather('../output/elevation_nldas')

all = pd.concat(all)
df = pd.merge(all,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])
''''''
df_grouped = df.groupby(['latitude','longitude','window']).count()['storm_id'].reset_index()

df_grouped['elev_cat'] = df.groupby(['latitude','longitude','window']).max().reset_index()['elevation_category']

df_grouped['region'] = df.groupby(['latitude','longitude','window']).max().reset_index()['region']
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

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the axes array to iterate over

# Determine global x and y limits
x_min = df_grouped['window'].min()
x_max = df_grouped['window'].max()
y_min = df_grouped['storm_id'].min()
y_max = df_grouped['storm_id'].max()


for idx, region in enumerate(regions):
    plot = df_grouped[df_grouped.region == region]
    sns.boxplot(data=plot, x='window', y='storm_id', hue='elev_cat', ax=axes[idx])
    axes[idx].set_title(f'Region {region}')

    axes[idx].set_ylim(y_min, y_max)
plt.tight_layout()
plt.show()