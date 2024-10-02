#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
elev = pd.read_feather('../output/'+'nldas'+'_elev')

#%%
# Create a 4x4 subplot
regions = [
    [12, 13, 14, 15],  # Top row
    [8, 9, 10, 11],    # Second row
    [4, 5, 6, 7],      # Third row
    [0, 1, 2, 3]       # Bottom row
]

# Flatten the regions list
regions = [region for sublist in regions for region in sublist]

fig, axes = plt.subplots(4, 4, figsize=(15*.5, 12*.5))
axes = axes.flat

for idx, region in enumerate(regions):
    plot = elev[elev.region==region]
    plot = plot.groupby(['latitude','longitude']).max().to_xarray()

    axes[idx].contourf(plot.longitude,plot.latitude,plot.elevation.values,add_colorbar=False,cmap='terrain',vmin = elev.elevation.min(), vmax = elev.elevation.max())

    axes[idx].text(0.8, 0.8, f'{region}', horizontalalignment='center', 
            verticalalignment='center', transform=axes[idx].transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')
    if idx in range(4):  # Only keep x-axis labels for the first row (idx 0-3)
        axes[idx].set_xticklabels([])
    
    if idx not in range(12, 16):  # Only keep x-axis labels for the last row (idx 12-15)
        axes[idx].set_xticklabels([])

    if (idx % 4) !=0:  # Only keep y-axis labels for the last column (idx 3, 7, 11, 15)
        axes[idx].set_yticklabels([])
    
    axes[idx].tick_params(left=False, bottom=False)

plt.subplots_adjust(wspace=.0, hspace=0.0)
plt.show()
fig.savefig("../figures_output/region_map.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
