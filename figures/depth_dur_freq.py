#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
#elev = pd.read_feather('../output/elevation')
elev = pd.read_feather('../output/elevation_nldas')
df_all_years = []
for window in (1,3,12,24):
    print(window)
    all_files = glob.glob('../output/duration_'+str(window)+'/*')
    files_nldas = glob.glob('../output/duration_'+str(window)+'/*nldas'+'*')
    aorc_files = [item for item in all_files if item not in files_nldas]

    for file in files_nldas:

        df = pd.read_feather(file)
        df = pd.merge(df,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

        for region in range(0,16):
            for elev_cat in df.elevation_category.unique():
                test = df[(df.region == region) & (df.elevation_category == elev_cat)]

                test = test.groupby(['quant']).agg(list)['storm_id'].reset_index()

                test['events'] = [len(np.unique(test.storm_id[i])) for i in range(len(test))]

                test['region'] = region
                test['elev_cat'] = elev_cat
                test['duration'] = window
                test['year'] = df.year[0]
                df_all_years.append(test)


# %%
df = pd.concat(df_all_years)

test = df.groupby(['quant', 'region',
       'elev_cat', 'duration']).sum().reset_index()
#test['freq'] = test.events/(2023-1979)
test['freq'] = test.events/(2013-2002)
test = test[test.elev_cat.isin(['High','Low'])]

test = test.drop(columns=['events', 'year'])
df_pivot = test.pivot_table(index=['quant', 'region', 'duration'], 
                          columns='elev_cat', 
                          values='freq')
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

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.5, 16*.5), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the axes array to iterate over

# Plot each region
for idx, region in enumerate(regions):
    plot = df_result[df_result.region == region]

    # For the first plot, keep the legend, and for others, set `legend=False`
    sns.lineplot(
        data=plot,
        y="difference",
        x="quant",
        hue="duration",  # The hue for the legend
        ax=axes[idx],
        palette='tab10',
        legend=False if idx > 0 else 'full'  # Show legend only on the first subplot
    )

    #axes[idx].set_ylim(-1, 1)
    axes[idx].text(0.5, 0.9, f'Region {region}', horizontalalignment='center', 
                   verticalalignment='center', transform=axes[idx].transAxes, 
                   bbox=dict(facecolor='white', alpha=0.5))

    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')
    axes[idx].axhline(0, color='gray', linestyle='--')



# Add shared axis labels
fig.text(-.01, .1, 'frequency (events/year) difference (high-low)', ha='center', fontsize=16, rotation='vertical')
fig.text(0.5, -.01, 'quantile of max annual', ha='center', fontsize=16)

# Adjust layout
plt.tight_layout()  # Adjust the layout to make space for the legend
plt.show()
# %%
