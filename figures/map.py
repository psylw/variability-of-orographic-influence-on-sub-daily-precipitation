#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
#%%
elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude','longitude']).max().elevation.to_xarray()


#%%
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

# Plot the elevation contours
fig = plt.figure(figsize=(10, 8))
plt.contourf(elev.longitude, elev.latitude, elev, cmap='terrain', alpha=0.7)
plt.colorbar(label="Elevation (m)", fraction=0.03, orientation='horizontal', pad=0.08)

# Define custom text positions for each HUC shape
custom_positions = {
    10: (0.75, 0.75),  
    11: (0.62, 0.42),  
    13: (0.5, 0.15),  
    14: (0.05, 0.7),  
}

for shape in [10, 11, 13, 14]:
    # Load the shapefile
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    gdf = gpd.read_file(shapefile_path)

    # Plot the shapefile on top of the contour plot
    ax = plt.gca()  # Get the current axes
    gdf.plot(ax=ax, edgecolor='black', facecolor='none')  # Overlay shapefile with transparent fill

    # Extract the 'name' attribute (assuming it's a valid column)
    gdf_name = gdf['name'].iloc[0] if 'name' in gdf.columns else 'Unknown'

    # Add text with the shape number and gdf.name at the specified position
    pos_x, pos_y = custom_positions[shape]
    plt.text(
        pos_x, pos_y,
        f"HUC2 id: {shape}\n{gdf_name[:-7]}",
        transform=ax.transAxes,
        fontsize=12,
        color='blue',
        ha='left',
        va='top'
    )

# Load station data and rename columns
df = pd.read_feather("../output/coag_1")
df = df.groupby(['latitude','longitude']).max().reset_index()

# Plot the stations as red points
plt.scatter(df.longitude, df.latitude, c='red', label='CoAgMET stations',edgecolor='white')

# Set longitude and latitude limits
plt.xlim(elev.longitude.min(), elev.longitude.max())
plt.ylim(elev.latitude.min(), elev.latitude.max())

# Finalize the plot
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper left')
plt.show()
fig.savefig("../figures_output/map.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
