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
plt.colorbar(label="Elevation (m)", fraction=0.03, orientation='horizontal', pad=0.05)

# Define custom text positions for each HUC shape
custom_positions = {
    10: (0.8, 0.8),  
    11: (0.76, 0.39),  
    13: (0.5, 0.2),  
    14: (0.03, 0.7),  
}
name = {
    10: 'Missouri',  
    11: 'Arkansas',  
    13: 'Rio Grande',  
    14: 'Colorado',  
}

cities = {'City': ['Denver', 'Colorado Springs', 'Fort Collins', 'Grand Junction', 'pueblo'],
          'latitude': [39.7392, 38.8339, 40.5853, 39.0639,38.2544],
          'longitude': [-104.9903, -104.8214, -105.0844, -108.5506,-104.6091]}

cities_df = pd.DataFrame(cities)

radar = {'name':['kcys','kftg','kpux','kala','kgjx'],
         'latitude':[41.17,39.8, 38.47,37.46,39.05],
         'longitude':[-104.82, -104.56,-104.2,-105.86,-108.23]}
radar_df = pd.DataFrame(radar)
plt.scatter(cities_df.longitude,cities_df.latitude,
           s = 120, facecolors='red',marker="^",label='city',edgecolors='black')

plt.scatter(radar_df.longitude,radar_df.latitude,
           s =150,facecolors='cornflowerblue',marker='o',label='radar',edgecolors='black')
import textwrap

for shape in [10, 11, 13, 14]:
    # Load the shapefile
    shapefile_path = f"../data/huc2/WBD_{shape}_HU2_Shape/WBDHU2.shp"
    gdf = gpd.read_file(shapefile_path)

    # Plot the shapefile on top of the contour plot
    ax = plt.gca()  # Get the current axes
    gdf.plot(ax=ax, edgecolor='blue', facecolor='none')  # Overlay shapefile with transparent fill

    # Extract the 'name' attribute (assuming it's a valid column)
    gdf_name = name[shape]


    # Add text with the shape number and gdf.name at the specified position
    pos_x, pos_y = custom_positions[shape]
    plt.text(
        pos_x, pos_y,
        f"HUC2 id: {shape}\n{gdf_name}",
        transform=ax.transAxes,
        fontsize=12,
        color='blue',
        ha='left',
        va='top'
    )
gdf = gpd.read_file("../data/Colorado_Mtn_Ranges/ranges.shp")
num_shapes = len(gdf)
colors = plt.cm.Dark2(np.linspace(0, 1, num_shapes))
reordered_indices = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11]  # Manually adjusted
reordered_colors = [colors[i % len(colors)] for i in reordered_indices]

# Plot the shapefile on top of the contour plot
ax = plt.gca()  # Get the current axes
#gdf.plot(ax=ax, edgecolor='pink', facecolor='none',hatch='///') 
max_length = 10

for j, (idx, row) in enumerate(gdf.iterrows()):
        gdf.iloc[[idx]].plot(ax=ax, edgecolor=reordered_colors[j], facecolor='none', hatch='///')

        name = gdf.iloc[[idx]].name.values[0]
        name = "\n".join(textwrap.wrap(name, width=11))

        ax.text(gdf.iloc[[idx]].centroid.x-.2,gdf.iloc[[idx]].centroid.y,name,color=reordered_colors[j],bbox=dict(facecolor='white', alpha=0.75,edgecolor='none',pad=.3))

# Load station data and rename columns
df = pd.read_feather("../output/coag_1")
df = df.groupby(['latitude','longitude']).max().reset_index()

# Plot the stations as red points
#plt.scatter(df.longitude, df.latitude, c='red', label='CoAgMET stations',edgecolor='white')

# Set longitude and latitude limits
plt.xlim(elev.longitude.min(), elev.longitude.max())
plt.ylim(elev.latitude.min(), elev.latitude.max())

# Finalize the plot
plt.xlabel("")
plt.ylabel("")
plt.legend(loc='upper left')
plt.show()
fig.savefig("../figures_output/f01.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')

# %%
