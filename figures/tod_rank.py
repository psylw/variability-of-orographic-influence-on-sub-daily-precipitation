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
    df = pd.read_feather('../output/'+name+'_rank'+years)

    df['time_utc'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')

    # Convert from UTC to Denver time (Mountain Time)
    df['time_denver'] = df['time_utc'].dt.tz_convert('America/Denver')

    df['hour'] = df.time_denver.dt.hour
 
    df = pd.merge(df,elev,on=['latitude','longitude']).dropna()
    
    max_rank = df.groupby(['window','year','region']).max().precip_ranked.reset_index().rename(columns={'precip_ranked':'max_rank'})

    df = pd.merge(df,max_rank,on=['window','year','region'])

    df['relative_rank']=df.precip_ranked/df.max_rank

    df['binned_values'] = pd.cut(df['relative_rank'], bins=[0,.25,.5,.75,1],labels=[1,2,3,4])

    test = df.groupby(['window','binned_values','region','elevation_category','hour']).count().latitude.reset_index().rename(columns={'latitude':'cnt'})

    num_region_elevbin = df.groupby(['window','region','elevation_category']).count().latitude.rename('totalsample').reset_index()

    test = pd.merge(test,num_region_elevbin ,on=['window','region','elevation_category'])
    test['freq'] = test.cnt/test.totalsample

    test['dataset'] = name

    return test

'''df1= open_data('aorc','nldas','2016')
df2 = open_data('nldas','nldas','2016')
df3 = open_data('conus','nldas','2016')
df4 = open_data('mrms','nldas','2016')
#df_conus2 = open_data('conus','conus','all')
#df_conus2['dataset'] = 'conus2'

df = pd.concat([df1,df2,df3,df4])
df = df.dropna()
df['elevation_category'] = df['elevation_category'].astype('string')'''
df = open_data('conus','conus','all')
df['elevation_category'] = df['elevation_category'].astype('string')

#%%
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]


#y_labels = ['Low', 'Lower Middle', 'Upper Middle', 'High']
y_labels = ['Low',  'High']
# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

fig, axes = plt.subplots(4, 4, figsize=(15*.6, 12*.6), sharex=True, sharey=True)
ax = axes.flat

for idx, region in enumerate(regions):
    plot = df[(df.window==1)&(df.region==region)&(df.binned_values==4)]


    plot['binary'] = plot['elevation_category'].map({'Low':0, 'Lower Middle':1, 'Upper Middle':2, 'High':3})
    plot = plot[plot.binary.isin([0,3])]

    plot = plot.groupby(['binary','hour']).max().to_xarray().freq
    
    im = plot.plot(ax=ax[idx], edgecolor=None, vmin=0,vmax=.02,add_colorbar=False, rasterized=True,cmap='Blues')

    #plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
    plt.yticks(ticks=[0,3], labels=y_labels)
    plt.xticks(ticks=[0,6,12,18,23], labels=[0,6,12,18,23])

    ax[idx].set_xlabel('')
    ax[idx].set_ylabel('')
    ax[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=ax[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))


cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=.01)
cbar.set_label('frequency')
fig.supxlabel('hour of day (MDT)', x=0.5, y=0.05)
#plt.tight_layout()
fig.savefig("../figures_output/tod_rank.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
