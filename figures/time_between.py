#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
def open_data(name):
    elev = pd.read_feather('../output/'+'nldas'+'_elev')
    df = pd.read_feather('../output/time_between_'+name)
    df = pd.merge(df,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])
    df = df.groupby(['region','percent','elevation_category']).agg(list).dropna().reset_index()
    df['time_between'] = [np.concatenate(df.iloc[i]['time_between']) for i in df.index]
    df = df.explode('time_between').dropna()
    df = df.drop(columns=['latitude','longitude','year','threshold'])
    df['dataset'] = name
    return df
df1= open_data('aorc')
df1['dataset'] = 'AORC'
df2 = open_data('nldas')
df2['dataset'] = 'NLDAS-2'
df3 = open_data('conus')
df3['dataset'] = 'CONUS404'
df4 = open_data('mrms')
df4['dataset'] = 'MRMS'

df = pd.concat([df1,df2,df3,df4])
df['elevation_category'] = df['elevation_category'].astype('string')
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
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.65, 16*.65),sharex=True,sharey=True)
axes = axes.flatten()  # Flatten the axes array to iterate over
dataset_colors = {
'Low':sns.color_palette("colorblind")[0], 
'High':sns.color_palette("colorblind")[1], }
for idx, region in enumerate(regions):
    plot = df[(df.region == region)&(df['percent']==df['percent'].unique()[0])&(df.elevation_category.isin(['High','Low']))]

    scatter = sns.histplot(data=plot, x="time_between", hue="elevation_category",
    palette=dataset_colors,
    ax=axes[idx])

    axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
    verticalalignment='center', transform=axes[idx].transAxes, 
    bbox=dict(facecolor='white', alpha=0.5))
    axes[idx].set_xlabel('')

    axes[idx].set_ylabel('')


    scatter_handles, scatter_labels = scatter.get_legend_handles_labels()

    #scatter.get_legend().remove()
'''handles = scatter_handles
labels = scatter_labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-.08, 0.85), title=None,frameon=False)'''



plt.tight_layout()
plt.show()
#fig.savefig("../figures_output/elevvs_ann_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')


# %%
