from pathlib import Path
import argparse

from PIL import Image, ImageOps

def preprocess_img(img_path, size=224):
    with Image.open(img_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    width, height = image.size
    img_cropped = image.crop((0.25 * width, 0.5 * height, 0.75 * width, height))

    # Resize 
    scale = size / min(img_cropped.size)
    new_size = tuple(max(size, round(side * scale)) for side in img_cropped.size)
    resized_img = img_cropped.resize(new_size, Image.Resampling.LANCZOS)

    # Center crop
    left = (resized_img.width - size) // 2
    top = (resized_img.height - size) // 2
    prepared_img = resized_img.crop(
        (left, top, left + size, top + size)
    )

    return img_cropped, prepared_img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Did not find directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in extensions:
            continue

        _, prepared_image = preprocess_img(img_path)
        output_path = output_dir / f"{img_path.stem}.png"
        prepared_image.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

