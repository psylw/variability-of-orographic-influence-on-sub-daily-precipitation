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
def open_data(name,elevds,years):
    df_window = []
    for window in (1,3,12,24):
        elev = pd.read_feather('../output/'+elevds+'_elev')
        df = xr.open_dataset('../output/'+name+'_ann_max_px_'+years+'_window_'+str(window)+'.nc').median(dim='year').to_dataframe()

        df = pd.merge(df,elev,on=['latitude','longitude'])

        df['dataset'] = name
        df['window'] = window

        df_window.append(df)

    df_window = pd.concat(df_window)
    return df_window
#%%
df_aorc = open_data('aorc','nldas','2016')
df_nldas = open_data('nldas','nldas','2016')
df_conus = open_data('conus','nldas','2016')
df_conus2 = open_data('conus','conus','allyears')
df_conus2['dataset'] = 'conus2'

df = pd.concat([df_aorc,df_nldas,df_conus,df_conus2])
df = df.dropna()
#%%
'''df['category'] = pd.cut(df['elevation'], bins=np.arange(df.elevation.min(),df.elevation.max(),500), labels=np.arange(df.elevation.min(),df.elevation.max(),500)[0:-1])'''

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
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18*.75, 16*.75))
axes = axes.flatten()  # Flatten the axes array to iterate over

dataset_colors = {
    'aorc': sns.color_palette("colorblind")[0],
    'nldas': sns.color_palette("colorblind")[1],
    'conus': sns.color_palette("colorblind")[2],
    'conus2': 'darkgreen'
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
fig.savefig("../figures_output/elevvs_ann_max.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
reg = []
for region in range(0,16):
    plot = df[(df.region == region)]

    for window in (1,3,12,24):
        plot_window = plot[plot.window==window]
 
        for data in ['aorc','nldas','conus','conus2']:
            #data = 'aorc'
            plot_ds = plot_window[plot_window['dataset']== data]
            x=plot_ds.elevation
            y=plot_ds.accum
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            '''            reg.append({'region':region,
                'win'+str(window)+data+'slope':slope,
                'win'+str(window)+data+'r_value':r_value,
                })'''

            
            reg.append({'region':region,
                        'window':window,
                        'dataset':data,
                        'slope':slope,
                        'r_value':r_value**2

                })
                
reg = pd.DataFrame(reg)
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
    'aorc': sns.color_palette("colorblind")[0],
    'nldas': sns.color_palette("colorblind")[1],
    'conus': sns.color_palette("colorblind")[2],
    'conus2': 'darkgreen'
}

window_styles = {
    1: ('o', 'solid'),  # Window 1: circle marker and dashed line
    24: ('s', 'dashed'), 
}

# Plot each region

for idx, region in enumerate(regions):
    plot = reg[(reg.region == region)]
    

    sns.lineplot(
        data=plot,
        x="window",
        y="slope",
        hue="dataset",
        palette=dataset_colors,
        legend=False,
        ax=axes[idx],  # Customize the error bars
    )
    ax2 = axes[idx].twinx()
    sns.scatterplot(
        data=plot,
        x="window",
        y="r_value",
        hue="dataset",
        palette=dataset_colors,
        legend=False,
        ax=ax2,  # Customize the error bars
    )
    if (idx % 4) < 3:
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_yticks([])

    ax2.set_ylim(0,1)
    #axes[idx].set_ylim(-.005,.015)
    axes[idx].axhline(y=0,color='black',linewidth=.5,alpha=.5,linestyle='--')

    axes[idx].set_xticks([1,3,12,24])
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')
    ax2.set_ylabel('')
    axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=axes[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

fig.supxlabel('accumulation window, hr')
fig.supylabel('slope mm/m')
fig.text(1, 0.5, 'R-squared', va='center', ha='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.show()
fig.savefig("../figures_output/slopes.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
