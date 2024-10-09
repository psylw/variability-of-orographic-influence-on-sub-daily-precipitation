#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
def open_data(name):
    elev = pd.read_feather('../output/'+'conus'+'_elev')
    cell_area = pd.read_feather('../output/cell_area'+'conus')
    df_all_years = []

    window = 1
    files = glob.glob('../output/duration_'+str(window)+'/*'+name+'*lower*')
    
    for file in files:
        df = pd.read_feather(file)
        df = pd.merge(df,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

        for region in range(0,16):
            test = df[df.region==region]

            test = test.groupby(['storm_id','elevation_category','quant']).count().latitude.reset_index()
            test = test.rename(columns={'latitude':'area'})

            test['year'] = df.year[0]
            test['dataset'] = name
            test['dur_above'] = df[df.region==region].groupby(['storm_id','elevation_category','quant']).mean().accum.values
            test['region'] = region
            df_all_years.append(test)

    df = pd.concat(df_all_years)
    df = pd.merge(df,cell_area,on=['region'])
    df['area'] = df.area*df.cell_area

    df = df[df.area>0]
    df['dur_area'] = df.area*df.dur_above
    return df

# %%
df = open_data('conus_all')
df['dataset'] = 'CONUS404 1980-2022'
#%%
df1= open_data('aorc')
df2 = open_data('nldas')
df3 = open_data('conus')

df = pd.concat([df1,df2,df3])


# %%
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20*.5, 16*.5),sharex=True,sharey=True )
axes = axes.flatten()  # Flatten the axes array to iterate over


for idx, region in enumerate(regions):

        plot = df[(df.region == region)]
        plot = plot[plot.elevation_category.isin(['High','Low'])]
        plot['elevation_category'] = plot['elevation_category'].astype('object')

        scatter = sns.lineplot(data = plot, x = 'quant',y='area',style='elevation_category',hue='dataset', ax=axes[idx], palette={'CONUS404 1980-2022': 'darkgreen'})
        #axes[idx].set_yscale('log')
        #axes[idx].set_ylim(0,100)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
        scatter_handles2, scatter_labels2 = scatter.get_legend_handles_labels()

        scatter.get_legend().remove()

        axes[idx].text(0.9, 0.9, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=axes[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))
handles = scatter_handles2
labels = scatter_labels2
fig.legend([handles[3],handles[4]], [ 'Low', 'High'], loc='center left', bbox_to_anchor=(-0.1, 0.9), title=None,frameon=False)
# Add one shared x-axis label at the bottom middle
fig.text(-.01,.5, 'area', ha='center', fontsize=16, rotation='vertical')
fig.text(0.5, -.01, 'percentage of median ann max', ha='center', fontsize=16)
plt.tight_layout()
plt.show()
fig.savefig("../figures_output/area.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%