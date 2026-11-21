import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "outputs" / "manifest.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "manifest_split.csv"

EARTH_RADIUS_M = 6_371_000
DISTANCE_M = 100
SEED = 42
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
CLASS_IDS = ("0", "1", "2")


def assign_spatial_groups(rows):
    coordinates = np.array([
        [float(row["latitude"]), float(row["longitude"])]
        for row in rows
    ])
    # Haversine expects coordinates and distance in radians.
    group_labels = DBSCAN(
        eps=DISTANCE_M / EARTH_RADIUS_M,
        min_samples=1,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(np.radians(coordinates))

    for row, group_label in zip(rows, group_labels):
        row["group_id"] = f"geo_{group_label}"


def split_rows(rows):
    labels = np.array([row["class_id"] for row in rows])
    groups = [row["group_id"] for row in rows]
    if set(labels) != set(CLASS_IDS):
        raise ValueError("The manifest must contain class IDs 0, 1, and 2.")

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = [
        held_out
        for _, held_out in splitter.split(np.zeros(len(rows)), labels, groups)
    ]

    # Keep both class and source proportions close to the full dataset.
    sources = np.array([row.get("source", "") for row in rows])
    distinct_sources = np.unique(sources)
    fold_counts = np.array([
        [
            len(fold),
            *(np.count_nonzero(labels[fold] == label) for label in CLASS_IDS),
            *(np.count_nonzero(sources[fold] == source) for source in distinct_sources),
        ]
        for fold in folds
    ])
    total_counts = fold_counts.sum(axis=0)
    target_counts = np.array(list(SPLIT_RATIOS.values()))[:, None] * total_counts

    best_pair = None
    best_error = float("inf")
    for val_fold, test_fold in combinations(range(len(folds)), 2):
        val_counts = fold_counts[val_fold]
        test_counts = fold_counts[test_fold]
        train_counts = total_counts - val_counts - test_counts
        counts = np.array([train_counts, val_counts, test_counts])
        if (counts[:, 1:1 + len(CLASS_IDS)] == 0).any():
            continue
        error = np.square((counts - target_counts) / target_counts).sum()
        if error < best_error:
            best_pair = val_fold, test_fold
            best_error = error

    if best_pair is None:
        raise ValueError("Could not place all three classes in each split.")

    for row in rows:
        row["split"] = "train"
    for split, fold_index in zip(("validation", "test"), best_pair):
        for row_index in folds[fold_index]:
            rows[row_index]["split"] = split


def save_manifest(rows):
    with NamedTemporaryFile(
        dir=OUTPUT_PATH.parent, mode="w", encoding="utf-8", newline="", delete=False
    ) as file:
        temporary_path = Path(file.name)
        try:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        except Exception:
            file.close()
            temporary_path.unlink(missing_ok=True)
            raise

    try:
        temporary_path.replace(OUTPUT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    # Prefer the combined manifest so additions and their metadata are kept.
    input_path = OUTPUT_PATH if OUTPUT_PATH.is_file() else INPUT_PATH
    with input_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("The manifest must contain rows with unique image IDs.")
    rows.sort(key=lambda row: row["image_id"])

    populated_groups = [bool(row["group_id"].strip()) for row in rows]
    if not any(populated_groups):
        assign_spatial_groups(rows)
    elif not all(populated_groups):
        raise ValueError("Group IDs must be either present for all rows or empty for all.")

    split_rows(rows)
    save_manifest(rows)

    print("Saved:", OUTPUT_PATH)
    print("Total images:", len(rows))
    print("Total groups:", len({row["group_id"] for row in rows}))
    split_counts = Counter(row["split"] for row in rows)
    for split in SPLIT_RATIOS:
        count = split_counts[split]
        print(f"{split}: {count} ({count / len(rows):.2%})")


if __name__ == "__main__":
    main()
