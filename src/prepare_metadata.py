import csv
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_DIR / "dataset" / "streetSurfaceVis_v1_0.csv"

CLASSES = ("unpaved", "concrete", "sett")
CLASS_TO_ID = {
    label: class_id
    for class_id, label in enumerate(CLASSES)
}

IMAGE_DIR = PROJECT_DIR / "outputs" / "preprocessed"
MANIFEST_PATH = PROJECT_DIR / "outputs" / "manifest.csv"

with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
    rows = [row for row in csv.DictReader(file)
            if row["surface_type"] in CLASSES]

def check():
    print("Total images:", len(rows))
    print("Images counts each classes:", Counter(row["surface_type"] for row in rows))

    image_id = rows[0]["mapillary_image_id"]

    original_path = (
        PROJECT_DIR / "dataset" / "s_1024" / f"{image_id}.jpg"
    )

    processed_path = (
        PROJECT_DIR / "outputs" / "preprocessed" / f"{image_id}.png"
    )

    print("Original images exist:", original_path.is_file())
    print("Preprocessed images exist:", processed_path.is_file())

def create_manifest(rows):
    manifest = []
    seen_ids = set()
    split_mapping = {
        "True": "development",
        "False": "test",
    }

    for row in rows:
        image_id = row["mapillary_image_id"]
        label = row["surface_type"]
        image_path = IMAGE_DIR / f"{image_id}.png"


        if image_id in seen_ids:
            raise ValueError(f"Image_id in seen_id: {image_id}")

        if not image_path.is_file():
            raise FileNotFoundError(f"Did not find image: {image_path}")

        split_value = row["train"].strip()

        if split_value not in split_mapping:
            raise ValueError(f"Inapproriate train flags: {split_value}")

        manifest.append({
            "image_id": image_id,
            "image_path": image_path.relative_to(PROJECT_DIR).as_posix(),
            "label": label,
            "class_id": CLASS_TO_ID[label],
            "split": split_mapping[split_value],
            "longitude": row["longitude"],
            "latitude": row["latitude"],
            "captured_at": row["captured_at"],
            "group_id": "",
        })
        seen_ids.add(image_id)

    return manifest

def save_manifest(manifest):
    if not manifest:
        raise ValueError("No images found for creating manifest.")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(manifest[0]),
        )
        writer.writeheader()
        writer.writerows(manifest)

def main():
    manifest = create_manifest(rows)
    save_manifest(manifest)

    print("Saved:", MANIFEST_PATH)
    print("Total images:", len(manifest))
    print(
        "Images counts according to set and classes",
        Counter((item["split"], item["label"]) for item in manifest))

if __name__ == "__main__":
    main()
    
