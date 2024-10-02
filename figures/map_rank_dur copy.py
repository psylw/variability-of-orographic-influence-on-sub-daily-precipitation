#%%
import pandas as pd
import xarray as xr
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#%%
def open_data(name,elevds,years):
    elev = pd.read_feather('../output/'+elevds+'_elev')
    ds = xr.open_dataset('../output/'+name+'_rank'+years+'.nc')
    df = ds.to_dataframe().reset_index()
    df = pd.merge(df,elev,on=['latitude','longitude']).dropna()
    
    max_rank = df.groupby(['xarray_index','year','region']).max().precip_ranked.reset_index().rename(columns={'precip_ranked':'max_rank'})
    df = pd.merge(df,max_rank,on=['xarray_index','year','region'])

    df['relative_rank']=df.precip_ranked/df.max_rank

    df['binned_values'] = pd.cut(df['relative_rank'], bins=4, labels=[1, 2, 3, 4])

    test = df.groupby(['xarray_index','binned_values','latitude','longitude']).count().year.reset_index().rename(columns={'year':'cnt'})

    num_region_elevbin = df.groupby(['xarray_index','latitude','longitude']).count().year.rename('totalsample').reset_index()

    test = pd.merge(test,num_region_elevbin ,on=['xarray_index','latitude','longitude'])
    test['freq'] = test.cnt/test.totalsample

    test['dataset'] = name

    return test

df1= open_data('aorc','nldas','2016')
df2 = open_data('nldas','nldas','2016')
df3 = open_data('conus','nldas','2016')
df_conus2 = open_data('conus','conus','all')
df_conus2['dataset'] = 'conus2'

df = pd.concat([df1,df2,df3,df_conus2])
df = df.dropna()

#%%
ds1 = df1.groupby(['xarray_index','binned_values','latitude','longitude']).max().to_xarray()
ds2 = df2.groupby(['xarray_index','binned_values','latitude','longitude']).max().to_xarray()
ds3 = df3.groupby(['xarray_index','binned_values','latitude','longitude']).max().to_xarray()
ds_conus2 = df_conus2.groupby(['xarray_index','binned_values','latitude','longitude']).max().to_xarray()

#%%
xarray_list = []
for dataset in [ds1,ds3,ds2,ds_conus2]:

    xarray_list.append(dataset.sel(xarray_index=0,binned_values=4).freq)
    xarray_list.append(dataset.sel(xarray_index=3,binned_values=4).freq)
    xarray_list.append(dataset.sel(xarray_index=3,binned_values=4).freq-dataset.sel(xarray_index=0,binned_values=4).freq)

#%%
fig, axes = plt.subplots(4,3, figsize=(12, 16), sharex=True, sharey=True)
row_labels = ['aorc', 'conus', 'nldas','conus2']
# Loop through each xarray and corresponding subplot
for i, ax in enumerate(axes.flat):
    if (i % 3) < 2:
        im = xarray_list[i].plot(ax=ax,vmin=0,vmax=1, add_colorbar=False)
    else:
        im = xarray_list[i].plot(ax=ax,vmin=-.2,vmax=.2, add_colorbar=False,cmap='coolwarm')
    ax.set_xlabel('')
    ax.set_ylabel('')
    if i == 0:
        ax.set_title('1hr high int freq')
    elif i == 1:
        ax.set_title('24hr high int freq')
    elif i == 2:
        ax.set_title('diff')
    else:
        ax.set_title('')
for row in range(3):
    fig.text(-.01, 0.85 - row * 0.3, row_labels[row], va='center', ha='center', rotation='vertical', fontsize=12)

plt.subplots_adjust(wspace=.5, hspace=0.1)
# Adjust layout to avoid overlapping plots
plt.tight_layout()

# Show the plot
plt.show()
fig.savefig("../figures_output/map_diff.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
