
#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress
#%%
# compare median max precip at different thresholds at different windows
def open_data(name,elevds,years,window):
    df_window = []
    for window in window:
        elev = pd.read_feather('../output/'+elevds+'_elev')
        df = xr.open_dataset('../output/'+name+'_ann_max_px_'+years+'_window_'+str(window)+'.nc').to_dataframe().reset_index()

        df = pd.merge(df,elev,on=['latitude','longitude'])

        df['dataset'] = name
        df['window'] = window

        df_window.append(df)

    df_window = pd.concat(df_window)
    return df_window
#%%
df_aorc = open_data('aorc','nldas','2016',(1,3,12,24))
df_aorc['dataset'] = 'AORC 2016-2022'
df_nldas = open_data('nldas','nldas','2016',(1,3,12,24))
df_nldas['dataset'] = 'NLDAS-2 2016-2022'
df_conus = open_data('conus','nldas','2016',(1,3,12,24))
df_conus['dataset'] = 'CONUS404 2016-2022'
#df_mrms = open_data('mrms_nldas','nldas','2016',(1*2,5*2,15*2,1,3,12,24))
#df_mrms = df_mrms.drop(columns=['step','heightAboveSea'])
#df_mrms['window'] = df_mrms['window']
#df_mrms['dataset'] = 'MRMS 2016-2022'

df_conus2 = open_data('conus','conus','allyears',(1,3,12,24))
df_conus2['dataset'] = 'CONUS404 1980-2022'

df = pd.concat([df_aorc,df_nldas,df_conus])

#%%
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]
dataset_colors = {
'AORC 2016-2022':sns.color_palette("colorblind")[0], 
'NLDAS-2 2016-2022':sns.color_palette("colorblind")[1], 
'CONUS404 2016-2022':sns.color_palette("colorblind")[2],
       'MRMS 2016-2022':sns.color_palette("colorblind")[3], 
       'CONUS404 1980-2022':'darkgreen', 
       'MRMS 2016-2022 sub-hr':'darkred'
}

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.6, 16*.6))
axes = axes.flatten()  # Flatten the axes array to iterate over

for idx, region in enumerate(regions):
    plot1 = df[(df.region == region)&(df.window==1)].dropna()
    plot2 = df_conus2[(df_conus2.region == region)&(df_conus2.window==1)].dropna()

    plot1['e'] = pd.cut(plot1['elevation'], 50)
    plot1['e']  = plot1['e'].apply(lambda x: x.mid)
    plot2['e'] = pd.cut(plot2['elevation'], 50)
    plot2['e']  = plot2['e'].apply(lambda x: x.mid)
    ''''''
    line = sns.lineplot(
        data=plot1,
        x="e",
        y = 'accum',
        palette= dataset_colors,
        hue="dataset",

        errorbar= ("pi", 90), 
        estimator='median',
        ax=axes[idx]
    )
    line2 = sns.lineplot(
        data=plot2,
        x="e",
        y = 'accum',
        palette= dataset_colors,
        hue="dataset",
        dashes=(2, 2),
        errorbar= ("pi", 90), 
        estimator='median',
        ax=axes[idx]
    )

    #axes[idx].set_xlim(0,50)
    #axes[idx].set_xticks([0,5,10,20,40])
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')
    axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=axes[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

    line_handles, line_labels = line.get_legend_handles_labels()
    line.get_legend().remove()


handles = line_handles
labels = line_labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.15, 0.9), title=None,frameon=False)

fig.supxlabel('elevation, m')
fig.supylabel('intensity, mm/hr')

plt.tight_layout()
plt.show()

fig.savefig("../figures_output/24hrspread.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
