"""Kiểm tra ảnh raw được tách nền và chuẩn hóa đúng hình học."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from src import prepare_images as preparation


class PrepareImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "raw"
        self.input_dir.mkdir()
        self.path = self.input_dir / "fruit.png"
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((60, 50, 79, 89), fill=(50, 100, 20))
        image.save(self.path)
        self.mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(self.mask).rectangle((60, 50, 79, 89), fill=255)

    def test_crop_preserves_aspect_color_and_mask_alignment(self):
        session = object()
        with patch.object(preparation, "remove", return_value=self.mask) as remove:
            image, mask = preparation.prepare_image(self.path, session)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (224, 224))
        self.assertEqual(mask.getbbox(), (64, 16, 160, 208))
        self.assertEqual(image.getpixel((112, 112)), (50, 100, 20))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual({value for _, value in mask.getcolors()}, {0, 255})
        self.assertIs(remove.call_args.kwargs["session"], session)

    def test_threshold_discards_faint_background(self):
        self.mask.putpixel((0, 0), 127)
        with patch.object(preparation, "remove", return_value=self.mask):
            _, mask = preparation.prepare_image(self.path, object())
        self.assertEqual(mask.getbbox(), (64, 16, 160, 208))

    def test_empty_mask_is_rejected(self):
        with patch.object(preparation, "remove", return_value=Image.new("L", (100, 100))):
            with self.assertRaisesRegex(ValueError, "Không tìm thấy"):
                preparation.prepare_image(self.path, object())

    def test_batch_saves_pairs_reuses_session_and_refuses_overwrite(self):
        second = self.input_dir / "sub" / "other.png"
        second.parent.mkdir()
        second.write_bytes(self.path.read_bytes())
        output = self.root / "processed"
        with (
            patch.object(preparation, "INPUT_DIR", self.input_dir),
            patch.object(preparation, "OUTPUT_DIR", output),
            patch.object(preparation, "new_session") as new_session,
            patch.object(preparation, "remove", return_value=self.mask) as remove,
            patch("sys.stdout"),
        ):
            self.assertEqual(preparation.main(), 0)
            new_session.assert_called_once()
            self.assertEqual(remove.call_count, 2)
            for call in remove.call_args_list:
                self.assertIs(call.kwargs["session"], new_session.return_value)
            for relative in (Path("fruit.png.png"), Path("other.png.png")):
                self.assertTrue((output / "images" / relative).is_file())
                self.assertTrue((output / "masks" / relative).is_file())
            self.assertFalse((output / "images" / "sub").exists())
            self.assertFalse((output / "masks" / "sub").exists())
            with self.assertRaises(FileExistsError):
                preparation.main()

    def test_duplicate_names_are_rejected_before_processing(self):
        second = self.input_dir / "sub" / "fruit.png"
        second.parent.mkdir()
        second.write_bytes(self.path.read_bytes())
        with (
            patch.object(preparation, "INPUT_DIR", self.input_dir),
            patch.object(preparation, "OUTPUT_DIR", self.root / "processed"),
            patch.object(preparation, "new_session") as new_session,
        ):
            with self.assertRaisesRegex(ValueError, "trùng tên"):
                preparation.main()
            new_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
