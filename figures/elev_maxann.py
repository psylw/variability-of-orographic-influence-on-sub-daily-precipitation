# map frequency of events above threshold
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
        df = xr.open_dataset('../output/'+name+'_ann_max_px_'+years+'_window_'+str(window)+'.nc').median(dim='year').to_dataframe()

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

df = pd.concat([df_aorc,df_nldas,df_conus,df_conus2])


# %%
reg = []
for region in range(0,16):
    plot = df[(df.region == region)]

    for data in ['AORC 2016-2022', 'NLDAS-2 2016-2022', 'CONUS404 2016-2022',
       'CONUS404 1980-2022']:


        for window in (1,3,12,24):
            plot_window = plot[plot.window==window]
            #data = 'aorc'
            plot_ds = plot_window[plot_window['dataset']== data].dropna()
            x=plot_ds.elevation
            y=plot_ds.accum
            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            reg.append({'region':region,
                        'window':window,
                        'dataset':data,
                        'slope':slope,
                        'r_value':r_value**2
                })
                
reg = pd.DataFrame(reg)
'''reg2 = []
for region in range(0,16):
    plot = df[(df.region == region)]

    data='MRMS 2016-2022'

    for window in (2/60,10/60,30/60):
        plot_window = plot[plot.window==window*60]
        #data = 'aorc'
        plot_ds = plot_window[plot_window['dataset']== data].dropna()
        x=plot_ds.elevation
        y=plot_ds.accum
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        reg2.append({'region':region,
                    'window':window,
                    'dataset':data,
                    'slope':slope,
                    'r_value':r_value**2
            })
                
reg2 = pd.DataFrame(reg2)
reg2['window'] = reg2['window']*60
reg2['dataset'] = 'MRMS 2016-2022 sub-hr'
reg = pd.concat([reg,reg2])'''

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
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.6, 16*.6),sharex=True,sharey=True)
axes = axes.flatten()  # Flatten the axes array to iterate over
dataset_colors = {
'AORC 2016-2022':sns.color_palette("colorblind")[0], 
'NLDAS-2 2016-2022':sns.color_palette("colorblind")[1], 
'CONUS404 2016-2022':sns.color_palette("colorblind")[2],
       'MRMS 2016-2022':sns.color_palette("colorblind")[3], 
       'CONUS404 1980-2022':'darkgreen', 
       'MRMS 2016-2022 sub-hr':'darkred'
}

marker_dict = {
'AORC 2016-2022':sns.color_palette("colorblind")[0], 
'NLDAS-2 2016-2022':sns.color_palette("colorblind")[1], 
'CONUS404 2016-2022':sns.color_palette("colorblind")[2],
       'MRMS 2016-2022':sns.color_palette("colorblind")[3], 
       'CONUS404 1980-2022':'darkgreen', 
       'MRMS 2016-2022 sub-hr':'darkred'
}
linestyle_dict = {
'AORC 2016-2022':sns.color_palette("colorblind")[0], 
'NLDAS-2 2016-2022':sns.color_palette("colorblind")[1], 
'CONUS404 2016-2022':sns.color_palette("colorblind")[2],
       'MRMS 2016-2022':sns.color_palette("colorblind")[3], 
       'CONUS404 1980-2022':'darkgreen', 
       'MRMS 2016-2022 sub-hr':'darkred'
}

for idx, region in enumerate(regions):
    plot1 = reg[(reg.region == region)&(reg.dataset != 'MRMS 2016-2022 sub-hr')]
    
    line = sns.lineplot(
        data=plot1,
        x="window",
        y="slope",
        hue="dataset",
        palette=dataset_colors,
        style='dataset',  
        ax=axes[idx],
    )
    ax2 = axes[idx].twinx()

    scatter = sns.scatterplot(
        data=plot1,
        x="window",
        y="r_value",
        hue="dataset",
        style='dataset',
        palette=dataset_colors,
        ax=ax2,  
    )

    #ax2_0 = axes[idx].twiny()
    #plot2 = reg[(reg.region == region)&(reg.dataset == 'MRMS 2016-2022 sub-hr')]
    '''    line2 = sns.lineplot(
            data=plot2,
            x="window",
            y="slope",
            hue="dataset",
            palette=dataset_colors,
            style='dataset',  
            ax=ax2_0,
        )
    ax3 = ax2_0.twinx()
    scatter2 = sns.scatterplot(
            data=plot2,
            x="window",
            y="r_value",
            hue="dataset",
            style='dataset',
            palette=dataset_colors,
            ax=ax3,  
        )'''

    #ax3.set_xlabel('')
    #ax3.set_ylabel('')
    #ax3.set_yticks([])
    if (idx % 4) < 3:
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_yticks([])

    ax2.set_ylim(0,1)
    #ax3.set_ylim(0,1)
    axes[idx].axhline(y=0,color='black',linewidth=.5,alpha=.5,linestyle='-')

    axes[idx].axhline(y=0.004,color='black',linewidth=.5,alpha=.5,linestyle='-')
    axes[idx].axhline(y=-0.004,color='black',linewidth=.5,alpha=.5,linestyle='-')

    axes[idx].axhline(y=0.002,color='black',linewidth=.5,alpha=.5,linestyle='-')
    axes[idx].axhline(y=-0.002,color='black',linewidth=.5,alpha=.5,linestyle='-')


    #ax2_0.set_xticks([2,10,30])
    axes[idx].set_ylim(-.005,.005)
    axes[idx].set_xticks([1,3,12,24])
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')
    #ax2_0.set_xlabel('')

    ax2.set_ylabel('')
    axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=axes[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

    scatter_handles, scatter_labels = scatter.get_legend_handles_labels()
    line_handles, line_labels = line.get_legend_handles_labels()
    scatter.get_legend().remove()
    line.get_legend().remove()

    #scatter_handles2, scatter_labels2 = scatter2.get_legend_handles_labels()
    #line_handles2, line_labels2 = line2.get_legend_handles_labels()
    #scatter2.get_legend().remove()
    #line2.get_legend().remove()

handles = scatter_handles + line_handles 
labels = scatter_labels + line_labels 
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.16, 0.88), title=None,frameon=False)

fig.supxlabel('accumulation window, hr')
fig.supylabel('slope, mm/hr-m')
fig.text(1, 0.5, 'R-squared', va='center', ha='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.show()
fig.savefig("../figures_output/slopes.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
'''dataset_colors = {
    'AORC 2016-2022': sns.color_palette("colorblind")[0],
    'NLDAS-2 2016-2022': sns.color_palette("colorblind")[1],
    'conus': sns.color_palette("colorblind")[2],
    'conus2': 'darkgreen',
    'mrms_nldas': sns.color_palette("colorblind")[3]
}

marker_dict = {
    'aorc': 'o', 'nldas': 'o', 'conus': 'o', 'conus2': 'X','mrms_nldas':'o'
}
linestyle_dict = {
    'conus2': (5, 5),
    'aorc': '',
    'nldas': '',
    'conus': '',
    'mrms_nldas': '',
}'''

# Plot each region
#%%
'''df['category'] = pd.cut(df['elevation'], bins=np.arange(df.elevation.min(),df.elevation.max(),500), labels=np.arange(df.elevation.min(),df.elevation.max(),500)[0:-1])'''

#%%
'''regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.75, 16*.75))
axes = axes.flatten()  # Flatten the axes array to iterate over

dataset_colors = {
    'aorc': sns.color_palette("colorblind")[0],
    'nldas': sns.color_palette("colorblind")[1],
    'conus': sns.color_palette("colorblind")[2],
    'conus2': 'pink'
}

window_styles = {
    1: ('o', 'solid'),  # Window 1: circle marker and dashed line
    24: ('s', 'dashed'), 
}

# Plot each region

for idx, region in enumerate(regions):
    plot = df[(df.region == region)]

    plot['category'] = pd.cut(plot['elevation'], bins=np.arange(plot.elevation.min(),plot.elevation.max(),20), labels=np.arange(plot.elevation.min(),plot.elevation.max(),20)[0:-1])

    for window in (1,24):
        plot_window = plot[plot.window==window]
        marker, linestyle = window_styles[window]
        # Create the plot

        sns.lineplot(
            data=plot_window.dropna(),
            x="category",
            y="accum",
            hue="dataset",
            palette=dataset_colors,
            err_style="bars",
            linewidth=0,
            err_kws={'elinewidth':2},
            alpha=.3,
            legend=False,
            ax=axes[idx],  # Customize the error bars
        )
        axes[idx].set(xlabel=None, ylabel=None)

        for data in ['aorc','nldas','conus','conus2']:
            plot_ds = plot_window[plot_window['dataset']== data]
            x=plot_ds.elevation
            y=plot_ds.accum
            slope, intercept, r_value, p_value, std_err = linregress(x, y)  # Degree 1 for linear fit
            
            y_fit = slope * np.sort(x.unique()) + intercept
            axes[idx].plot(np.sort(x.unique()), y_fit, label=f'Linear Fit: y={slope:.2f}x \nR²={r_value**2:.2f}, p={p_value:.2e}', color=dataset_colors[data],linestyle=linestyle)
        #axes[idx].legend()
        axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
        verticalalignment='center', transform=axes[idx].transAxes, 
        bbox=dict(facecolor='white', alpha=0.5))



fig.supxlabel('elevation, m')
fig.supylabel('accumulation, mm')
plt.tight_layout()
plt.show()
fig.savefig("../figures_output/elevvs_ann_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')'''
