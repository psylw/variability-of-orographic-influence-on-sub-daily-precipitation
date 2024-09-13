#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
elev = pd.read_feather('../output/elevation')
df_all_years = []
for window in (1,3,12,24):
    print(window)
    files = glob.glob('../output/duration_'+str(window)+'/*')

    for file in files:

        df = pd.read_feather(file)
        df = pd.merge(df,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

        for region in range(0,16):
            test = df[df.region==region]

            test = test.groupby(['storm_id','elevation_category','quant']).count().latitude.reset_index()

            test['region'] = region
            test['duration'] = window
            test['year'] = df.year[0]

            df_all_years.append(test)

# %%
df = pd.concat(df_all_years)
df = df[df.latitude>0]

test = df.groupby(['quant', 'region',
       'elevation_category', 'duration']).median().latitude.reset_index()


test = test[test.elevation_category.isin(['High','Low'])]

test['elevation_category'] = test['elevation_category'].astype('object')

df_pivot = test.pivot_table(index=['quant', 'region', 'duration'], 
                          columns='elevation_category', 
                          values='latitude')
df_pivot['difference'] = df_pivot['High'] - df_pivot['Low']

# Reset the index to get a flat DataFrame
df_result = df_pivot.reset_index().dropna()
# %%
# Define the regions in reverse row order
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20*.5, 16*.5), sharex=True)
axes = axes.flatten()  # Flatten the axes array to iterate over


for idx, region in enumerate(regions):
    plot = df_result[df_result.region == region]

    sns.lineplot(
    data=plot,
    y="difference",  # Use the new offset x values
    x="quant",
    #style="elevation_category",
    hue="duration", ax=axes[idx],palette='tab10',legend=False if idx > 0 else 'full')

    
    ##axes[idx].set_xlim(0,3)
    axes[idx].text(0.5, 0.9, f'Region {region}', horizontalalignment='center', 
                verticalalignment='center', transform=axes[idx].transAxes, 
                bbox=dict(facecolor='white', alpha=0.5))
    
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')
    axes[idx].axhline(0, color='gray', linestyle='--')


# Add one shared x-axis label at the bottom middle
fig.text(-.01,.5, 'area', ha='center', fontsize=16, rotation='vertical')
fig.text(0.5, -.01, 'quantile of max annual', ha='center', fontsize=16)
plt.tight_layout()
plt.show()


# %%
