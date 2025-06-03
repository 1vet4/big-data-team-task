# STEP 0: Import Required Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim
import pandas as pd
from sklearn.cluster import DBSCAN
import folium
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np


# STEP 1: Initialize Spark
spark = SparkSession.builder \
    .appName("AIS Port Detection") \
    .getOrCreate()

# STEP 2: Load and Filter AIS Data
file_path = "aisdk-2025-03-11.csv"

df_raw = spark.read.csv(file_path, header=True, inferSchema=True)

df_filtered = df_raw \
    .filter(col("Type of mobile") == "Class A") \
    .filter(col("Latitude").isNotNull() & col("Longitude").isNotNull()) \
    .filter((col("Latitude") >= -90) & (col("Latitude") <= 90)) \
    .filter((col("Longitude") >= -180) & (col("Longitude") <= 180)) \
    .filter(
        (col("SOG") < 1.0) |
        (lower(trim(col("Navigational status"))) == "moored")
    ) \
    .select("MMSI", "Latitude", "Longitude", "SOG", "# Timestamp", "Navigational status")

# STEP 3: Sample Down for Clustering
df_sample = df_filtered.sample(fraction=0.02).toPandas()

# STEP 4: Apply DBSCAN Clustering
# Drop any rows with missing coordinates
df_sample = df_sample.dropna(subset=["Latitude", "Longitude"])

# Convert degrees to radians
coords_rad = np.radians(df_sample[["Latitude", "Longitude"]])

# eps in radians = km / Earth radius (6371 km)
# we choose radius of 10km to include ships that are waiting to be let into port
db = DBSCAN(eps=10 / 6371,  # ~10km
            min_samples=8,
            metric='haversine')
db.fit(coords_rad)

df_sample["cluster"] = db.labels_

# Filter out ships that are moving
df_sample = df_sample[df_sample["SOG"] < 0.5]  # knots

# STEP 5: Analyze Clusters
cluster_sizes = df_sample["cluster"].value_counts()
valid_clusters = cluster_sizes[cluster_sizes >= 50].index # filter for bigger ports
df_sample = df_sample[df_sample["cluster"].isin(valid_clusters)]

print("\nTop Detected Ports by Point Count:")
print(cluster_sizes.head(10))

# STEP 6: Visualize with Folium
map_center = [df_sample["Latitude"].mean(), df_sample["Longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=6)

for cluster_id in cluster_sizes.index:
    cluster_data = df_sample[df_sample["cluster"] == cluster_id]
    lat = cluster_data["Latitude"].mean()
    lon = cluster_data["Longitude"].mean()

    # Check for NaN coordinates
    if pd.isna(lat) or pd.isna(lon):
        continue  # skip bad clusters

    size = len(cluster_data)
    folium.CircleMarker(
        location=[lat, lon],
        radius=min(size / 10, 50),
        popup=f"Port {cluster_id} ({size} points)",
        color="blue",
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

m.save("pyspark_ports_map.html")
print("🗺️ Port map saved as 'pyspark_ports_map.html'.")

# STEP 7: Optional – Plot with matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(df_sample["Longitude"], df_sample["Latitude"],
            c=df_sample["cluster"], cmap="tab20", s=10, alpha=0.6)
plt.title("Detected Ports (DBSCAN clustering)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()
