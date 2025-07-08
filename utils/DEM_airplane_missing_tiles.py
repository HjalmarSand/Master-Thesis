import geopandas as gpd
import matplotlib.pyplot as plt

# The path to your .shp file. geopandas will automatically find the other files.
shapefile_path = "./DSM_Data/missing_points/ytmodell_saknade_ytor_indexrutor.shp" 

try:
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # --- Let's see what's inside ---
    
    # Print the first 5 rows of the attribute table (.dbf content)
    print("Attribute Data:")
    print(gdf.head())

    # Print the Coordinate Reference System (.prj content)
    print("\nCoordinate Reference System (CRS):")
    print(gdf.crs)

    # Get the type of geometry (.shp content)
    print(f"\nGeometry Type: {gdf.geom_type.unique()}")

    # --- Let's plot it ---
    print("\nPlotting the shapefile data...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax)
    plt.title("Map of Features from Shapefile")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Error reading shapefile: {e}")