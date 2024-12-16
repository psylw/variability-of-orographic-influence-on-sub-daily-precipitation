#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde

#%%

df_conus = pd.read_feather('../output/conus_ann_max').groupby(['latitude','longitude','year']).max()

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df_conus,df_elev,on=['latitude','longitude'])
#%%
def categorize_elevation(group):
    group['e_cat'] = pd.qcut(group['elevation'], q=100, labels=range(0,100))
    return group

# Apply categorization within each group
df = df.groupby('huc2').apply(categorize_elevation)

df['e_cat'] = df['e_cat'].astype('float')
# %%
fig, axes = plt.subplots(2, 2, figsize=(15*.7, 10*.7))
axes = axes.flatten()

for i, h in enumerate(np.sort(df.huc2.unique())):
    plot = df[df.huc2==h]

    labels = plot[['elevation','e_cat']].groupby('e_cat').apply(lambda x: x.mean()).elevation.astype('int')

    scatter = plot[['e_cat','accum_24hr']].groupby('e_cat').quantile(.9).accum_24hr.reset_index()

    t=pd.merge(plot,scatter.rename(columns={'accum_24hr':'9q'}),on='e_cat')

    result = t[t['accum_24hr'] > t['9q']]

    sns.scatterplot(data = result, x='e_cat',y='accum_24hr',s=20,ax=axes[i],label='24 hour')

    scatter = plot[['e_cat','accum_1hr']].groupby('e_cat').quantile(.9).accum_1hr.reset_index()

    t=pd.merge(plot,scatter.rename(columns={'accum_1hr':'9q'}),on='e_cat')

    result = t[t['accum_1hr'] > t['9q']]

    #sns.scatterplot(data = result, x='e_cat',y='accum_1hr',s=20,ax=axes[i],label='1 hour')
    sns.lineplot(data = result, x='e_cat',y='accum_1hr',s=20,ax=axes[i],label='1 hour')
    axes[i].set_xticks(ticks=labels.loc[[0,19,39,59,79,99]].index, labels=labels.loc[[0,19,39,59,79,99]].values)
    axes[i].set_title(f"HUC2: {h}", loc='right', pad=-100, fontsize=14)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].legend(loc='upper left')
    if i != 0:
        axes[i].legend().remove()

    #sns.kdeplot(data=selected,x='e_cat',y='accum_24hr',ax=axes[i])

fig.supxlabel("Elevation (m)", fontsize=14)
fig.supylabel("Precipitation Accumulation (mm)", fontsize=14)
# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

fig.savefig("../figures_output/conusall.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')