"""Ảnh raw → rembg → crop → căn giữa trên nền đen 224 × 224."""

import os
import sys
from pathlib import Path

from PIL import Image, ImageOps
from rembg import new_session, remove


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "raw_img"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "prepared_224"
MODEL = "u2netp"
SIZE = 224
PADDING = 16
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_image(image_path, session):
    """Tách nền ảnh raw, trả về ảnh RGB và mask đã chuẩn hóa."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    # Chỉ dự đoán mask; màu quả lấy từ ảnh gốc.
    mask = remove(image, session=session, only_mask=True).convert("L")
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Không tìm thấy vùng quả.")

    fruit = image.crop(bbox)
    mask = mask.crop(bbox)
    black = Image.new("RGB", fruit.size, (0, 0, 0))
    black.paste(fruit, (0, 0), mask)

    # Cạnh dài của quả bằng 192 pixel; giữ tỉ lệ hai chiều.
    scale = (SIZE - 2 * PADDING) / max(fruit.size)
    new_size = tuple(max(1, round(side * scale)) for side in fruit.size)
    fruit = black.resize(new_size, Image.Resampling.LANCZOS)
    mask = mask.resize(new_size, Image.Resampling.NEAREST)

    position = ((SIZE - new_size[0]) // 2, (SIZE - new_size[1]) // 2)
    result = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    result_mask = Image.new("L", (SIZE, SIZE), 0)
    result.paste(fruit, position)
    result_mask.paste(mask, position)
    return result, result_mask


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError("Thư mục đầu ra đã tồn tại. Đổi OUTPUT_DIR để tránh ghi đè.")
    paths = sorted(path for path in INPUT_DIR.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"Không tìm thấy ảnh trong {INPUT_DIR}")


    cache = PROJECT_DIR / "outputs" / ".rembg"
    os.environ.setdefault("REMBG_HOME", str(cache))
    os.environ.setdefault("U2NET_HOME", str(cache))
    print(f"Đang nạp {MODEL} trên CPU...", flush=True)
    session = new_session(MODEL, providers=["CPUExecutionProvider"])

    succeeded = 0
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        try:
            image, mask = prepare_image(path, session)
            name = path.name + ".png"
            for folder, result in (("images", image), ("masks", mask)):
                destination = OUTPUT_DIR / folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                result.save(destination)
            succeeded += 1
        except (OSError, ValueError, RuntimeError) as error:
            print(f"LỖI: {error}")

    print(f"Thành công: {succeeded}/{len(paths)}. Đầu ra: {OUTPUT_DIR}")
    return 0 if succeeded == len(paths) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
