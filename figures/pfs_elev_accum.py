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
df_conus = pd.read_feather('../output/conus_new_ann_max').groupby(['latitude','longitude','year']).max().drop(columns='season').groupby(['latitude','longitude']).quantile(.9).reset_index()
df_conus['dataset'] = 'conus'
df = pd.concat([df_atlas,df_reps,df_conus])

df_elev = pd.read_feather('../output/conus_elev')

df = pd.merge(df,df_elev,on=['latitude','longitude'])
#%%
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
#%%
df['norm_1'] = df1['norm_1']
df['norm_24'] = df2['norm_24']
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
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

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

        # 2) Manually add regression lines with labels, again no local legend
        for dataset in subset['dataset'].unique():
            ds_plot = subset[subset['dataset'] == dataset]

            X = ds_plot['elevation'].values.reshape(-1,1) / 1000
            y = ds_plot[f"{window}hr accum"].values

            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            r2 = r2_score(y, model.predict(X))

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
        legend_entries.append((heading_artist, subplot_title))
        for handle, label in zip(h_sub, l_sub):
            legend_entries.append((handle, label))

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
final_handles, final_labels = zip(*legend_entries)

# Shift the subplots so the legend on the right has space
plt.subplots_adjust(right=0.65)

# Make one combined legend on the right
fig.legend(
    final_handles,
    final_labels,
    loc='center left',
    bbox_to_anchor=(.95, 0.5),  # shift further right if needed
    frameon=False
)

fig.supxlabel("Elevation (m)", fontsize=14, x=.5)
fig.supylabel("Precipitation Accumulation (mm)", fontsize=14)


plt.tight_layout()
plt.show()
fig.savefig("../figures_output/pfselevvsmag.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
