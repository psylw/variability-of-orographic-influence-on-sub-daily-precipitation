#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde
#%%
df_atlas = pd.read_feather('../output/atlas_14')
df_reps = pd.read_feather('../output/reps').drop(columns=['band_x', 'spatial_ref_x', 'band_y', 'spatial_ref_y'])
df_conus = pd.read_feather('../output/conus_ann_max').groupby(['latitude','longitude','year']).max().drop(columns='season').groupby(['latitude','longitude']).quantile(.9).reset_index()
df_conus['dataset'] = 'conus'
df = pd.concat([df_atlas,df_reps,df_conus])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])
# %%
bin_edges = range(int(df['elevation'].min()), int(df['elevation'].max()) + 100, 100)
# Create labels for the bins
bin_labels = bin_edges[:-1]

df['elevation_bin'] = pd.cut(df['elevation'], bins=bin_edges, labels=bin_labels, right=False)
df['elevation_bin'] = df.elevation_bin.astype('float')

df = df.rename(columns={'accum_1hr':'1hr accum','accum_24hr':'24hr accum'})
df['dataset'] = df['dataset'].replace({'atlas14':'ATLAS 14','refs':'REPS','conus':'CONUS404'})

dataset_colors = {'ATLAS 14':sns.color_palette("colorblind")[0], 'REPS':sns.color_palette("colorblind")[1], 'CONUS404':sns.color_palette("colorblind")[2]}

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Get unique huc2 values
unique_hucs = [10,11,13,14]

reg_values = []
# Create a 2x2 subplot grid
fig, axes = plt.subplots(4,1, figsize=(15*.8, 13*.8))

axes = axes.flatten()  # Flatten to easily iterate over the 2x2 grid
for window in ['1hr accum','24hr accum']:
    for i, h in enumerate(unique_hucs[:4]):  # Plot up to 4 HUC2 groups
        ax = axes[i]
        plot = df[df.huc2 == h].copy()
        
        # Create the pointplot using the numeric elevation_bin
        sns.pointplot(
            data=plot, 
            x='elevation_bin', 
            y=window, 
            hue='dataset',
            marker='',
            linestyle='none',
            
            capsize=0.2,
            errorbar=('pi', 50),
            err_kws={'linewidth': 1.5},
            legend=False,
            native_scale=True,
            palette= dataset_colors,
            ax=ax
        )
        
        # Plot linear regression for each dataset using the numeric elevation_bin with dashed lines
        for dataset in plot['dataset'].unique():
            dataset_plot = plot[plot['dataset'] == dataset]
            
            # Fit linear regression to get R²
            X = dataset_plot['elevation'].values.reshape(-1, 1)/1000
            y = dataset_plot[window].values
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            slope = model.coef_[0]
            reg_values.append([window,dataset,h,slope,r2])
            if window == '1hr accum':
                # Plot regression line with dashed style
                line=sns.regplot(
                    data=dataset_plot,
                    x='elevation',
                    y=window,
                    scatter=False,
                    ci=None,
                    color= dataset_colors[dataset],
                    line_kws={'linestyle': '--','alpha':.5},
                    ax=ax,
                    label=f"{dataset} 1hr,\n R²={r2:.2f}, slope={slope:.2f} mm/km"
                    #label=f"{dataset}"
                )

            else:
                                # Plot regression line with dashed style
                line=sns.regplot(
                    data=dataset_plot,
                    x='elevation',
                    y=window,
                    scatter=False,
                    ci=None,
                    color= dataset_colors[dataset],
                    line_kws={'linestyle': '-','alpha':.5},
                    ax=ax,
                    label=f"{dataset} 24hr,\n R²={r2:.2f}, slope={slope:.2f} mm/km"
                    #label=f"{dataset}"
                )


        # Set a subset of x-ticks to avoid clutter
    
        ax.set_title(f"HUC2: {h}", loc='right', pad=-100, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True)

        ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


fig.supxlabel("Elevation (m)", fontsize=14, x=.4)
fig.supylabel("Precipitation Accumulation (mm)", fontsize=14)
# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
fig.savefig("../figures_output/pfselevvsmag.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')