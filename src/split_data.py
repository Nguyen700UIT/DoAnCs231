import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "outputs" / "manifest.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "manifest_split.csv"


EARTH_RADIUS_M = 6_371_000
DISTANCE_M = 100

def split_development(rows):
    development_rows = [
        row for row in rows
        if row["split"] == "development"
    ]

    labels = [row["class_id"] for row in development_rows]
    groups = [row["group_id"] for row in development_rows]

    if any(not group.strip() for group in groups):
        raise ValueError("Exists image that doesn't have group_id")

    test_groups = {
        row["group_id"]
        for row in rows
        if row["split"] == "test"
    }

    if set(groups) & test_groups:
        raise ValueError("There are groups that are both in development and test")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    train_indices, val_indices = next(
        splitter.split(
            X=np.zeros(len(development_rows)),
            y=labels,
            groups=groups
        )
    )

    for index in train_indices:
        development_rows[index]["split"] = "train"

    for index in val_indices:
        development_rows[index]["split"] = "validation"

    expected_labels = set(labels)
    train_rows = [development_rows[i] for i in train_indices]
    val_rows = [development_rows[i] for i in val_indices]

    assert {row["class_id"] for row in train_rows} == expected_labels
    assert {row["class_id"] for row in val_rows} == expected_labels

    train_groups = {row["group_id"] for row in train_rows}
    val_groups = {row["group_id"] for row in val_rows}
    assert not (train_groups & val_groups)

    return rows


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

    split_development(rows)
    
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()