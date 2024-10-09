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


df1 = open_data('conus','conus','all').dropna()
#df2 = open_data('aorc','aorc','all').dropna()
#%%

elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude','longitude']).max().elevation.to_xarray()
#%%

df1 = df1.groupby(['xarray_index','binned_values','latitude','longitude']).max().to_xarray()
#df2 = df2.groupby(['xarray_index','binned_values','latitude','longitude']).max().to_xarray()

#%%
xarray_list = []
xarray_list.append(df1.sel(xarray_index=0,binned_values=4).freq)
xarray_list.append(df1.sel(xarray_index=3,binned_values=4).freq)
xarray_list.append(df1.sel(xarray_index=3,binned_values=4).freq-df1.sel(xarray_index=0,binned_values=4).freq)
'''xarray_list.append(df2.sel(xarray_index=0,binned_values=4).freq)
xarray_list.append(df2.sel(xarray_index=3,binned_values=4).freq)
xarray_list.append(df2.sel(xarray_index=3,binned_values=4).freq-df2.sel(xarray_index=0,binned_values=4).freq)'''


#%%
fig, axes = plt.subplots(1,3, figsize=(15*.65, 4.6*.65), sharex=True, sharey=True)

import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

cmap = get_cmap('Reds')
truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
    'truncatedReds', cmap(np.linspace(0.3, 1.0, 100))
)
cmap = get_cmap('coolwarm')


# Loop through each xarray and corresponding subplot
for i, ax in enumerate(axes.flat):


    if (i % 3) < 2 and (i // 3) == 0:
        
        im1 = ax.contourf(elev.longitude, elev.latitude, elev.values, cmap='terrain',alpha=.35)
        
        test = xarray_list[i].where(xarray_list[i]>.25).to_dataframe().dropna().reset_index()

        im=ax.scatter(test.longitude, test.latitude,s=1,c=test.freq,cmap=truncated_cmap,vmin=.25,vmax=1)

        cbar_ax_4th = fig.add_axes([1, 0.15, 0.01, .75]) 
        cbar = fig.colorbar(im, cax=cbar_ax_4th)
        cbar.set_label('frequency')
        cbar_ax_4th = fig.add_axes([0.36,-.05, .3, .05]) 
        cbar = fig.colorbar(im1, cax=cbar_ax_4th, orientation='horizontal')
        cbar.set_label('elevation, m')


    if (i % 3) == 2 and (i // 3) == 0:
        im1 = ax.contourf(elev.longitude, elev.latitude, elev.values, cmap='terrain',alpha=.35)

        test = xarray_list[i].to_dataframe().dropna().reset_index()
        test = test[(test.freq>.1)|(test.freq<-.1)]

        im = ax.scatter(test.longitude, test.latitude,s=1,c=test.freq,cmap='coolwarm',vmin=-.2,vmax=.2)

        cbar_ax_4th = fig.add_axes([1.07, 0.15, 0.01, .75]) 
        cbar = fig.colorbar(im, cax=cbar_ax_4th)
        cbar.set_label('24hr freq - 1hr freq')


    ax.set_xlabel('')
    ax.set_ylabel('')
    if i == 0:
        ax.set_title('1hr high int freq')
    elif i == 1:
        ax.set_title('24hr high int freq')
    elif i == 2:
        ax.set_title('24hr high int freq - 1hr high int freq')
    else:
        ax.set_title('')


plt.subplots_adjust(wspace=.5, hspace=0.1)
# Adjust layout to avoid overlapping plots
plt.tight_layout()

# Show the plot
plt.show()
fig.savefig("../figures_output/map_diff.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
