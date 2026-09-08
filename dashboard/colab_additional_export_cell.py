# Run this in Colab to export the two additional files the live dashboard needs

import pickle

# Add full historical soil moisture distribution (needed for soil_percentile
# in live_status.py — the earlier export only saved mean/std, not the raw
# per-month history a percentile calculation needs)
with open(f"{PROJECT_DIR}/kobo_climatology.pkl", "rb") as f:
    clim_data = pickle.load(f)

clim_data["soil_hist_by_month"] = {
    "soil_moisture": soil_df["soil_moisture"],
    "month": soil_df["month"],
}

with open(f"{PROJECT_DIR}/kobo_climatology.pkl", "wb") as f:
    pickle.dump(clim_data, f)

# Export the Kobo boundary as a standalone GeoJSON the live app can load
# without needing the full 40MB eth_admin3.geojson file
kobo_gdf.to_file(f"{PROJECT_DIR}/kobo_boundary.geojson", driver="GeoJSON")

print("Updated kobo_climatology.pkl and saved kobo_boundary.geojson")
print(os.listdir(PROJECT_DIR))
