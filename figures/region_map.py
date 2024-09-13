import matplotlib.pyplot as plt
import numpy as np

region = s.groupby(['latitude','longitude']).max().reset_index()[['latitude','longitude']]
# Create a 1D array of length 16 and reshape it to 4x4
array_1d = region.index.values.reshape((4,4))

# Define latitude and longitude coordinates for the grid
latitudes = region.latitude.unique()
longitudes = region.longitude.unique()

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data using imshow
cax = ax.imshow(data, cmap='viridis', origin='upper')

# Add value labels in the middle of each cell, with lat/lon coordinates
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        #lat_lon_label = f"({latitudes[i]}, {longitudes[j]})"
        value_label = str(data[i, j])
        ax.text(j, i, f"{value_label}", ha='center', va='center', color='white')

# Add a colorbar
fig.colorbar(cax)

# Set axis labels
ax.set_xticks(np.arange(len(longitudes)))
ax.set_xticklabels(longitudes)
ax.set_yticks(np.arange(len(latitudes)))
ax.set_yticklabels(latitudes)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Display the plot
plt.show()