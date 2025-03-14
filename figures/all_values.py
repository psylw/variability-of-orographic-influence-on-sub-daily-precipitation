#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde

#%%

df_conus = pd.read_feather('../output/conus_new_ann_max')


df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df_conus,df_elev,on=['latitude','longitude'])

bin_edges = range(int(df['elevation'].min()), int(df['elevation'].max()) + 500, 500)
# Create labels for the bins
bin_labels = bin_edges[:-1]

df['elevation_bin'] = pd.cut(df['elevation'], bins=bin_edges, labels=bin_labels, right=False)
df['elevation_bin'] = df.elevation_bin.astype('float')



fig, axes = plt.subplots(2, 2, figsize=(15*.7, 10*.7))
axes = axes.flatten()

for i, h in enumerate(np.sort(df.huc2.unique())):
    plot = df[df.huc2==h]

    scatter = plot[['elevation_bin','accum_1hr', 'season']].groupby(['elevation_bin','season']).quantile(.9).accum_1hr.reset_index()


    t=pd.merge(plot,scatter.rename(columns={'accum_1hr':'9q'}),on=['elevation_bin','season'])

    result = t[t['accum_1hr'] > t['9q']]

    sns.stripplot(data = result, x='elevation_bin',y='accum_1hr',hue='season',ax=axes[i],dodge=True,alpha=.25,legend=False,s=5)

    scatter = plot[['elevation_bin','accum_24hr', 'season']].groupby(['elevation_bin','season']).quantile(.9).accum_24hr.reset_index()

    t=pd.merge(plot,scatter.rename(columns={'accum_24hr':'9q'}),on=['elevation_bin','season'])

    result = t[t['accum_24hr'] > t['9q']]

    #sns.stripplot(data = result, x='elevation_bin',y='accum_24hr',hue='season',ax=axes[i],dodge=True,alpha=.25,marker="d",s=5)
    #sns.lineplot(data = result, x='e_cat',y='accum_1hr',s=20,ax=axes[i],label='1 hour')
    #axes[i].set_xticks(ticks=labels.loc[[0,19,39,59,79,99]].index, labels=labels.loc[[0,19,39,59,79,99]].values)
    axes[i].set_title(f"HUC2: {int(h)}", loc='right', pad=-100, fontsize=14)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    '''    axes[i].legend(loc='upper left')
    if i != 0:
        axes[i].legend().remove()'''


    #sns.kdeplot(data=selected,x='e_cat',y='accum_24hr',ax=axes[i])

fig.supxlabel("Elevation (m)", fontsize=14)
fig.supylabel("Precipitation Accumulation (mm)", fontsize=14)
# Adjust layout to prevent overlap
#plt.title(season)
plt.tight_layout()
plt.show()

#fig.savefig("../figures_output/conusall.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
