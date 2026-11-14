import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "outputs" / "manifest.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "manifest_grouped.csv"


EARTH_RADIUS_M = 6_371_000
DISTANCE_M = 100

def main():
    with INPUT_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    #Sort by image_id
    rows.sort(key=lambda row: row["image_id"])

    coordinates = np.array([
        [float(row["latitude"]), float(row["longitude"])]
        for row in rows
    ])
    #Distance between points on Earth can be calculate with haversine distance
    #Haversine distance requires coordinate to be radians.
    coordinates_rad = np.radians(coordinates)

    group_labels = DBSCAN(
        eps=DISTANCE_M / EARTH_RADIUS_M,
        min_samples=1,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coordinates_rad)


    for row, group_label in zip(rows, group_labels):
        row["group_id"] = f"geo_{group_label}"

        group_counts = Counter(row["group_id"] for row in rows)

    development_groups = {
        row["group_id"]
        for row in rows
        if row["split"] == "development"
    }
    test_groups = {
        row["group_id"]
        for row in rows
        if row["split"] == "test"
    }

    print("Total images:", len(rows))
    print("Total groups:", len(group_counts))
    print("5 largest groups:", group_counts.most_common(5))
    print("Groups that are both in development and test",len(development_groups & test_groups))


    with OUTPUT_PATH.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()