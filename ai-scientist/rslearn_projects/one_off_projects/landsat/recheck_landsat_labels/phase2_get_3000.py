"""Get a sample of 3000 vessel labels."""

import csv
import json
import os
import random
from datetime import datetime, timedelta, timezone

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import LocalFileAPI, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs

in_fname = "/multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/phase2_780k.csv"
count = 1000
out_ds_dir = "/multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/"
out_group = "phase2a"

candidates = []
with open(in_fname) as f:
    reader = csv.DictReader(f)
    for idx, csv_row in enumerate(reader):
        lon = float(csv_row["lon"])
        lat = float(csv_row["lat"])
        ts = datetime.fromisoformat(csv_row["timestamp"])
        if not ts.tzinfo:
            ts = ts.replace(tzinfo=timezone.utc)
        candidates.append((idx, lon, lat, ts))
selected = random.sample(candidates, 3000)

for idx, lon, lat, ts in selected:
    # Find appropriate UTM projection.
    dst_crs = get_utm_ups_crs(lon, lat)
    dst_projection = Projection(dst_crs, 15, -15)

    # Get window bounds centered at the vessel in dst_crs.
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(dst_projection)
    bounds = [
        int(dst_geom.shp.x) - 32,
        int(dst_geom.shp.y) - 32,
        int(dst_geom.shp.x) + 32,
        int(dst_geom.shp.y) + 32,
    ]

    time_range = (ts - timedelta(minutes=20), ts + timedelta(minutes=20))
    format_string = "%Y-%m-%dT%H:%M:%SZ"
    from_time = time_range[0].strftime(format_string)
    to_time = time_range[1].strftime(format_string)
    url = f"https://apps.sentinel-hub.com/eo-browser/?zoom=16&lat={lat}&lng={lon}&fromTime={from_time}&toTime={to_time}&datasetId=AWS_LOTL1&layerId=2_TRUE_COLOR_PANSHARPENED"

    window_name = f"vessel_{idx}"
    window_root = os.path.join(out_ds_dir, "windows", out_group, window_name)
    os.makedirs(window_root, exist_ok=True)
    window = Window(
        file_api=LocalFileAPI(window_root),
        group=out_group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    label_layer_dir = os.path.join(window_root, "layers", "label")
    os.makedirs(label_layer_dir, exist_ok=True)
    with open(os.path.join(label_layer_dir, "completed"), "w") as f:
        pass
    if not os.path.exists(os.path.join(label_layer_dir, "data.geojson")):
        with open(os.path.join(label_layer_dir, "data.geojson"), "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [[32, 32]],
                            },
                            "properties": {
                                "label": "unknown",
                                "url": url,
                                "idx": idx,
                                "lon": lon,
                                "lat": lat,
                                "ts": ts.isoformat(),
                            },
                        }
                    ],
                    "properties": window.projection.serialize(),
                },
                f,
            )
