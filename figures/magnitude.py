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
all_aorc = []
all_nldas = []
for window in (1,3,12,24):
    
    print(window)
    all_files = glob.glob('../output/duration_'+str(window)+'/*')
    files_nldas = glob.glob('../output/duration_'+str(window)+'/*nldas'+'*')
    aorc_files = [item for item in all_files if item not in files_nldas]
    '''    df_all_years_aorc = []
    for file in aorc_files:
        df = pd.read_feather(file)
        df = df[(df.quant==q)]
        df_all_years_aorc.append(df)

    df_all_years_aorc = pd.concat(df_all_years_aorc)
    df_all_years_aorc['window'] = window
    all_aorc.append(df_all_years_aorc)
    df_all_years_aorc.groupby(['latitude','longitude']).median().max_precip.to_xarray().plot()
    plt.show()'''


    df_all_years_nldas = []
    for file in files_nldas:
        df = pd.read_feather(file)
        df = df[(df.quant==q)]
        df_all_years_nldas.append(df)

    df_all_years_nldas = pd.concat(df_all_years_nldas)
    df_all_years_nldas['window'] = window
    all_nldas.append(df_all_years_nldas)
    df_all_years_nldas.groupby(['latitude','longitude']).median().max_precip.to_xarray().plot()
    plt.show()
#%%
# boxplot frequency of events above threshold, by window, hue elevation
'''elev_aorc = pd.read_feather('../output/elevation')

all_aorc = pd.concat(all_aorc)
df_aorc = pd.merge(all_aorc,elev_aorc[['latitude','longitude','elevation_category']],on=['latitude','longitude'])'''


elev_nldas = pd.read_feather('../output/elevation_nldas')

all_nldas = pd.concat(all_nldas)
df_nldas = pd.merge(all_nldas,elev_nldas[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

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
    plot = df_aorc[df_aorc.region == region]
    sns.boxplot(data=plot, x='window', y='max_precip', hue='elevation_category', ax=axes[idx])
    axes[idx].set_title(f'Region {region}')

plt.tight_layout()
plt.show()


#%%
def min_max_normalize(group):
    group['normalized_max_precip'] = (group['max_precip'] - group['max_precip'].min()) / (group['max_precip'].max() - group['max_precip'].min())
    return group

# Apply normalization based on region and window
df_nldas = df_nldas.groupby(['region', 'window']).apply(min_max_normalize)

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
    plot = df_nldas[df_nldas.region == region]
    sns.boxplot(data=plot, x='window', y='normalized_max_precip', hue='elevation_category', ax=axes[idx])
    axes[idx].set_title(f'Region {region}')

plt.tight_layout()
plt.show()
# %%
