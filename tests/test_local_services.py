import tempfile
import unittest
from pathlib import Path

from PIL import Image

from local_services import analyze_image, run_local_command


class LocalServicesTests(unittest.TestCase):
    def test_command_streams_output_and_exit_code(self):
        output = []
        return_code = run_local_command(
            "echo Kulu local command smoke test",
            str(Path.cwd()),
            output.append,
        )

        self.assertEqual(return_code, 0)
        self.assertIn("Kulu local command smoke test", "".join(output))

    def test_image_preview_stays_local_when_recognition_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "本地图片.png"
            Image.new("RGB", (32, 24), "white").save(image_path)

            result = analyze_image(str(image_path), use_ocr=False, use_objects=False)

        self.assertEqual(result["preview"].shape, (24, 32, 3))
        self.assertEqual(result["ocr"], [])
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["errors"], [])

    def test_invalid_image_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.png"
            invalid_path.write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not open this image"):
                analyze_image(str(invalid_path), use_ocr=False, use_objects=False)


if __name__ == "__main__":
    unittest.main()
