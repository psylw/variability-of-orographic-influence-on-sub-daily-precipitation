###############################################################################
# label each storm with transposition zone
###############################################################################
import geopandas as gpd

# open transposition zone shapefiles
shapefile_path = "../transposition_zones_shp/Transposition_Zones.shp"
gdf = gpd.read_file(shapefile_path)
# open storms, convert 10mm/hr threshold area to gpd

# check CRS

# plot to make sure everything in right place

# find zone with max intersection area
intersection = gpd.overlay(shapefile1, shapefile2, how='intersection')

# Calculate area of intersection
intersection_area = intersection.area.sum()