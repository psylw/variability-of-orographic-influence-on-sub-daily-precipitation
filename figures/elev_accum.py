

# show variability of precip for each elevation bin, plot linear regression

# %%
import re
import xarray as xr
import numpy as np
import glob
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
#import xesmf as xe
import seaborn as sns
import matplotlib.lines as mlines

#%%
df_mrms = pd.read_feather('../output/mrms_ann_max').drop(columns=['step','heightAboveSea'])
df_mrms1 = df_mrms.groupby(['latitude','longitude']).median().reset_index()
df_mrms1['dataset'] = 'MRMS'
df_aorc = pd.read_feather('../output/aorc_ann_max')
df_aorc1 = df_aorc.groupby(['latitude','longitude']).median().reset_index()
df_aorc1['dataset'] = 'AORC'
df_conus = pd.read_feather('../output/conus_new_ann_max')
df_conus_longer = df_conus[(df_conus.season=='JJA')].drop(columns='season')
df_conus_longer1 = df_conus_longer.groupby(['latitude','longitude']).median().reset_index()
df_conus_longer1['dataset'] = 'CONUS404 ER'
df_conus = df_conus[(df_conus.season=='JJA')&(df_conus.year>=2016)].drop(columns='season')
df_conus1 = df_conus.groupby(['latitude','longitude']).median().reset_index()
df_conus1['dataset'] = 'CONUS404'
df = pd.concat([df_mrms1,df_aorc1,df_conus1,df_conus_longer1])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])

window = 'accum_1hr'
df1 = df.copy()
df_max = df1[['huc2',window,'dataset']].groupby(['huc2','dataset']).max().rename(columns={window:'n_max'})

df_min = df1[['huc2',window,'dataset']].groupby(['huc2','dataset']).min().rename(columns={window:'n_min'})

df1 = pd.merge(df1,df_max,on=['huc2','dataset'])
df1 = pd.merge(df1,df_min,on=['huc2','dataset'])

df1['norm_1'] = (df1[window]-df1.n_min)/(df1.n_max-df1.n_min)

window = 'accum_24hr'
df2 = df.copy()
df_max = df2[['huc2',window,'dataset']].groupby(['huc2','dataset']).max().rename(columns={window:'n_max'})

df_min = df2[['huc2',window,'dataset']].groupby(['huc2','dataset']).min().rename(columns={window:'n_min'})

df2 = pd.merge(df2,df_max,on=['huc2','dataset'])
df2 = pd.merge(df2,df_min,on=['huc2','dataset'])

df2['norm_24'] = (df2[window]-df2.n_min)/(df2.n_max-df2.n_min)

df['norm_1'] = df1['norm_1']
df['norm_24'] = df2['norm_24']


df = df.rename(columns={'accum_1hr':'1hr accum','accum_24hr':'24hr accum'})
#%%


#%%
bin_edges = range(int(df['elevation'].min()), int(df['elevation'].max()) + 100, 100)
# Create labels for the bins
bin_labels = bin_edges[:-1]

df['elevation_bin'] = pd.cut(df['elevation'], bins=bin_edges, labels=bin_labels, right=False)
df['elevation_bin'] = df.elevation_bin.astype('float')

#%%
dataset_colors = {'MRMS':sns.color_palette("colorblind")[0], 'AORC':sns.color_palette("colorblind")[1], 'CONUS404':sns.color_palette("colorblind")[2], 'CONUS404 ER':'darkgreen'}

#%%

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from sklearn.feature_selection import f_regression
# Assume df has columns: huc2, elevation_bin, season, '1hr accum', '24hr accum', 'elevation', etc.
# and dataset_colors is a dict mapping e.g. 'DJF' -> some color, etc.
unique_hucs = [10, 11, 13, 14]
slope2_1hr_dict = {}
reg_values = []


# Create a 4×2 grid of subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12*.8, 10*.8))

# We'll store *all* legend entries (handles, labels) in one list
# for building a single legend with sub-headings.
legend_entries = []

for i, h in enumerate(unique_hucs):
    
    for j, window in enumerate([1, 24]):
        ax = axes[i, j]
        
        # Subset your data
        subset = df[df['huc2'] == h]

        # 1) Pointplot with no local legend
        sns.pointplot(
            data=subset,
            x='elevation_bin', 
            y=f"{window}hr accum", 
            hue='dataset',
            marker='',
            linestyle='none',
            capsize=0.2,
            errorbar=('pi', 50),
            err_kws={'linewidth': 1.5, 'alpha': 0.5},
            native_scale=True,
            palette=dataset_colors,
            legend=False,
            ax=ax
        )
        label_colours = []
        # 2) Manually add regression lines with labels, again no local legend
        for dataset in subset['dataset'].unique():
            ds_plot = subset[subset['dataset'] == dataset]

            X = ds_plot['elevation'].values.reshape(-1,1) / 1000
            y = ds_plot[f"{window}hr accum"].values

            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            r2 = r2_score(y, model.predict(X))
            _, p = f_regression(X,y)
            reg_values.append([p,dataset,window,h])
            lbl_colour = 'red' if p > 0.05 else 'black'
            label_colours.append(lbl_colour)

            # If 1hr, store slope2 for later comparison
            if window == 1:
                slope2 = LinearRegression().fit(X, ds_plot['norm_1']).coef_[0]
                slope2_1hr_dict[(dataset, h)] = slope2
                label = f"{dataset}, R²={r2:.2f}, slope={slope:.2f} ({slope2:.2f})"
                
            else:
                slope2_24 = LinearRegression().fit(X, ds_plot['norm_24']).coef_[0]
                slope2_1 = slope2_1hr_dict.get((dataset, h), None)
                if slope2_1 is not None and slope2_24 > slope2_1:
                    slope2_label = f"{slope2_24:.2f}"
                else:
                    slope2_label = rf"$\mathbf{{{slope2_24:.2f}}}$"
                label = rf"{dataset}, R²={r2:.2f}, slope={slope:.2f}, ({slope2_label})"
                


            # Actually plot the regression line with a label
            x_data = ds_plot['elevation'].values
            y_pred = model.predict(X)  # X in km
            ax.plot(
                x_data, y_pred,
                
                
                color=dataset_colors[dataset],
                label=label
            )

        # Remove any auto-legend Seaborn might have created
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # 3) Now gather handles/labels from this subplot
        h_sub, l_sub = ax.get_legend_handles_labels()

        # 4) Insert a dummy heading for this subplot
        #    We'll place a "blank" (invisible) line with the label as heading text
        subplot_title = f"HUC2: {h} ({window}hr)"
        heading_artist = Line2D([0], [0], color='none', label=subplot_title)

        # 5) Append the heading first, then this subplot’s lines
        #    (We do NOT deduplicate because we want each subplot’s lines under its heading,
        #     even if the same label appears in multiple subplots.)
        lbl_colour = 'black'
        legend_entries.append((heading_artist, subplot_title, lbl_colour))

        for handle, label, c in zip(h_sub, l_sub, label_colours):
            
            legend_entries.append((handle, label, c))

        # Cosmetic subplot settings

        ax.grid(True)
        ax.text(
            0.7,        # x in axes fraction
            1.1,        # y in axes fraction
            f"HUC2: {h} ({window}hr)",
            transform=ax.transAxes,  # so (x, y) are relative to the subplot
            ha='left',               # left-justify text
            va='top',                # align text by its top
            fontsize=12,             # adjust font size if needed
        )
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xlabel("")
        ax.set_ylabel("")

# Now we have a big list of (handle, label) pairs that includes 8 “headings,”
# plus all the lines from each subplot.

# Separate them out for fig.legend
final_handles, final_labels, colors = zip(*legend_entries)

# Shift the subplots so the legend on the right has space
plt.subplots_adjust(right=0.65)

# Make one combined legend on the right
leg = fig.legend(
    final_handles,
    final_labels,
    loc='center left',
    bbox_to_anchor=(.95, 0.5),  # shift further right if needed
    frameon=False,
    labelcolor = colors
)



fig.supxlabel("Elevation (m)", fontsize=14, x=.5)
fig.supylabel("Precipitation Accumulation (mm)", fontsize=14)


plt.tight_layout()
plt.show()


fig.savefig("../figures_output/f07.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
