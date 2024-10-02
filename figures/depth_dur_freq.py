#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
def open_data(name):
    elev = pd.read_feather('../output/'+name+'_elev')
    df_all_years = []
    thr = []
    for window in (1,3,12,24):
        files = glob.glob('../output/duration_'+str(window)+'/*'+name+'*')
        
        for file in files:
            threshold = pd.read_feather(file).groupby(['quant','region']).max().threshold.reset_index()
            threshold['duration'] = window
            thr.append(threshold)

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

    thr = pd.concat(thr)
    thr = thr.groupby(['quant','region','duration']).max().reset_index()
    df = pd.concat(df_all_years)

    df = df.groupby(['quant', 'region',
        'elev_cat', 'duration']).sum().reset_index()
    
    df = pd.merge(df,thr,on=['quant','region','duration'])

    df['freq'] = df.events/(21)
    df['dataset'] = name
    return df



# %%
df1= open_data('aorc')
df2 = open_data('nldas')
df3 = open_data('conus')

df = pd.concat([df1,df2,df3])


#test = test[test.elev_cat.isin(['High','Low'])]
#test = test[test.duration.isin([1,24])]
'''test = test.drop(columns=['events', 'year'])
df_pivot = test.pivot_table(index=['quant', 'region', 'duration'], 
                          columns='elev_cat', 
                          values='freq')
df_pivot['difference'] = df_pivot['High'] - df_pivot['Low']

# Reset the index to get a flat DataFrame
df_result = df_pivot.reset_index().dropna()'''

# %%
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.6, 16*.6), sharex=True,sharey=True)
axes = axes.flatten()  # Flatten the axes array to iterate over

dataset_colors = {
    'aorc': 'cornflowerblue',
    'nldas': sns.color_palette("tab10")[1],
    'conus': 'limegreen'
}

window_styles = {
    'High': 'o',  # Window 1: circle marker and dashed line
    'Low': '*',
       
}

offset = 3

for idx, region in enumerate(regions):
    for window in (1,24):
        if window == 24:
            plot = df[(df.region == region)&(df.duration==window)]
            plot = plot[plot.elev_cat.isin(['High','Low'])]
            plot['freq'] = plot.freq+3
        else:
            plot = df[(df.region == region)&(df.duration==window)]
            plot = plot[plot.elev_cat.isin(['High','Low'])]

        # For the first plot, keep the legend, and for others, set `legend=False`
        sns.lineplot(
            data=plot,
            y="freq",
            x="quant",
            hue="dataset",  # The hue for the legend
            ax=axes[idx],
            palette=dataset_colors,
            linewidth=0,
            #style='elev_cat',
            legend=False
            #legend=False if idx > 0 else 'full'  # Show legend only on the first subplot
        )

        sns.lineplot(
            data=plot,
            y="freq",
            x="quant",
            hue="dataset",  # The hue for the legend
            ax=axes[idx],
            palette=dataset_colors,
            style='elev_cat',
            legend=False
            #legend=False if idx > 0 else 'full'  # Show legend only on the first subplot
        )


        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        #axes[idx].set_ylim(-1, 1)
        axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
                    verticalalignment='center', transform=axes[idx].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.5))
        axes[idx].axhline(y=3,color='black',linewidth=.5)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
        axes[idx].set_xlim(0,.25)
        axes[idx].set_ylim(0,6)

        #ax2 = axes[idx].twinx()
        #ax2.set_yticks([0,1,2,3,4,5,6])
        axes[idx].set_yticklabels([0,1,2,0,1,2,3])


# Add shared axis labels
#fig.text(-.01, .5, 'frequency (events/year)', ha='center', fontsize=12, rotation='vertical')
#fig.text(0.5, -.01, 'quantile of max annual', ha='center', fontsize=12)
plt.subplots_adjust(wspace=.1, hspace=0.1)
# Adjust layout
#plt.tight_layout()  # Adjust the layout to make space for the legend
plt.show()

fig.savefig("../figures_output/depth_dur_freq.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')

# %%
'''regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.7, 16*.5), sharex=True)
axes = axes.flatten()  # Flatten the axes array to iterate over

offset_dict = {
    ('Low', 'aorc'): -0.3,
    ('Low', 'conus'): -0.2,
    ('Low', 'nldas'): -0.1,
    ('High', 'aorc'): 0.1,
    ('High', 'conus'): 0.2,
    ('High', 'nldas'): 0.3
}

# Plot each region
for idx, region in enumerate(regions):
    plot = df[(df.region == region)]

    plot = plot[plot.elev_cat.isin(['High','Low'])]
    plot['duration'] = [1, 2, 3, 4] * (len(plot) // 4) + [1, 2, 3, 4][:len(plot) % 4]

    # Apply the offsets to the 'duration' values
    plot['duration_offset'] = plot.apply(lambda row: row['duration'] + offset_dict[(row['elev_cat'], row['dataset'])], axis=1)

    # Plot using seaborn with offset 'duration' values
    sns.scatterplot(
        data=plot, x="duration_offset", y="freq",
        hue="elev_cat", style="dataset", 
        ax=axes[idx],legend=False

    )


    

# Adjust layout
plt.tight_layout()  # Adjust the layout to make space for the legend
plt.show()
# %%'''

