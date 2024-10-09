#%%
import pandas as pd
import xarray as xr
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
def open_data(name,elevds,years):
    elev = pd.read_feather('../output/'+elevds+'_elev')
    ds = pd.read_feather('../output/'+name+'_rank'+years)
    ds = ds.groupby(['longitude', 'latitude','window', 'year']).max().reset_index()

    df = pd.merge(ds,elev,on=['latitude','longitude']).dropna()
    df = df.rename(columns={'window':'xarray_index'})

    max_rank = df.groupby(['xarray_index','year','region']).max().precip_ranked.reset_index().rename(columns={'precip_ranked':'max_rank'})
    df = pd.merge(df,max_rank,on=['xarray_index','year','region'])

    df['relative_rank']=df.precip_ranked/df.max_rank

    df['binned_values'] = pd.cut(df['relative_rank'], bins=[0,.25,.5,.75,1],labels=[1,2,3,4])

    test = df.groupby(['xarray_index','binned_values','region','elevation_category']).count().latitude.reset_index().rename(columns={'latitude':'cnt'})

    num_region_elevbin = df.groupby(['xarray_index','region','elevation_category']).count().latitude.rename('totalsample').reset_index()

    test = pd.merge(test,num_region_elevbin ,on=['xarray_index','region','elevation_category'])
    test['freq'] = test.cnt/test.totalsample

    test['dataset'] = name

    return test

df1= open_data('aorc','nldas','2016')
df2 = open_data('nldas','nldas','2016')
df3 = open_data('conus','nldas','2016')
#df4 = open_data('mrms','nldas','2016')
#%%
elev = pd.read_feather('../output/'+'nldas'+'_elev')
ds = pd.read_feather('../output/'+'conus'+'_rank'+'2016.nc')
ds = ds.groupby(['longitude', 'latitude','window', 'year']).max().reset_index()

df = pd.merge(ds,elev,on=['latitude','longitude']).dropna()
df = df.rename(columns={'window':'xarray_index'})

max_rank = df.groupby(['xarray_index','year','region']).max().precip_ranked.reset_index().rename(columns={'precip_ranked':'max_rank'})
df = pd.merge(df,max_rank,on=['xarray_index','year','region'])

df['relative_rank']=df.precip_ranked/df.max_rank

df['binned_values'] = pd.cut(df['relative_rank'], bins=[0,.25,.5,.75,1],labels=[1,2,3,4])

test = df.groupby(['xarray_index','binned_values','region','elevation_category']).count().latitude.reset_index().rename(columns={'latitude':'cnt'})

num_region_elevbin = df.groupby(['xarray_index','region','elevation_category']).count().latitude.rename('totalsample').reset_index()

test = pd.merge(test,num_region_elevbin ,on=['xarray_index','region','elevation_category'])
test['freq'] = test.cnt/test.totalsample

test['dataset'] = 'conus'

#%%

#df_conus2 = open_data('conus','conus','all')
#df_conus2['dataset'] = 'conus2'

df = pd.concat([df1,df2,df3])
df = df.dropna()
df['elevation_category'] = df['elevation_category'].astype('string')

#%%
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

for idx, region in enumerate(regions):
    plot = df[(df.region==region)&(df.elevation_category.isin(['High','Low']))&(df.xarray_index.isin([1,24]))]

    plot = plot.rename(columns={'elevation_category':'elev cat', 'xarray_index':'window'})

    plot['window'] = plot['window'].replace({1: '1-hr', 24: '24-hr'})
    
    line = sns.lineplot(data = plot,
                       x='binned_values',
                       y='freq',
                       hue='elev cat',
                       style='window',
                       ax=axes[idx], 
                       #dashes=linestyles,
                       palette='tab10')

    axes[idx].set_ylim(.1,.4)
    axes[idx].set_xlabel(None)
    axes[idx].set_ylabel(None)

    axes[idx].set_xticks(ticks=list(np.arange(1,5,1)),labels=['0-.25','.25-.5','.5-.75','.75-1'])
    axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=axes[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

    line_handles, line_labels = line.get_legend_handles_labels()

    line.get_legend().remove()

handles = line_handles
labels = line_labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.08, 0.9), title=None,frameon=False)
fig.supxlabel('intensity rank')
fig.supylabel('frequency')
plt.tight_layout()

fig.savefig("../figures_output/rank_freq.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')

# %%
