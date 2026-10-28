"""Tạo hai nhãn độc lập từ thư mục ảnh đã chuẩn bị."""

import csv
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREPARED_DIR = PROJECT_DIR / "outputs" / "prepared_224"
LABELS = {
    "green_fresh": ("green", "fresh"),
    "green_withered": ("green", "withered"),
    "ripe_fresh": ("ripe", "fresh"),
    "ripe_withered": ("ripe", "withered"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_labels(prepared_dir):
    """Mỗi ảnh là một dòng; các góc chụp cùng quả giữ chung fruit_id."""
    rows = []
    for image in sorted((prepared_dir / "images").rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image.relative_to(prepared_dir / "images")
        if len(relative.parts) != 2 or relative.parts[0] not in LABELS:
            raise ValueError(f"Thư mục nhãn không hợp lệ: {relative}")
        match = re.match(r"^(fruit\d+)_", image.name)
        if match is None:
            raise ValueError(f"Tên ảnh phải bắt đầu bằng fruit<số>_: {image.name}")
        mask = prepared_dir / "masks" / relative
        if not mask.is_file():
            raise FileNotFoundError(f"Thiếu mask: {mask}")
        ripeness, freshness = LABELS[relative.parts[0]]
        rows.append({
            "fruit_id": match.group(1),
            "image_path": image.relative_to(PROJECT_DIR).as_posix(),
            "mask_path": mask.relative_to(PROJECT_DIR).as_posix(),
            "ripeness": ripeness,
            "freshness": freshness,
        })
    if not rows:
        raise ValueError(f"Không tìm thấy ảnh trong {prepared_dir / 'images'}")
    return rows


def main():
    rows = collect_labels(PREPARED_DIR)
    destination = PREPARED_DIR / "labels.csv"
    # Không ghi đè bảng có thể đã được chỉnh nhãn thủ công.
    with destination.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[1]))
        writer.writeheader()
        writer.writerows(rows)
    for labels, count in sorted(Counter((row["ripeness"], row["freshness"]) for row in rows).items()):
        print(f"{labels[0]} / {labels[1]}: {count}")
    print(destination)


if __name__ == "__main__":
    main()
