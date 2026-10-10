import tempfile
import unittest
from pathlib import Path

import yaml

from train_limited import CLASSES, prepare_data


class DatasetValidationTests(unittest.TestCase):
    def fixture(self, root):
        names = list(reversed(CLASSES))
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir()
            # Only the validator's label/order/leakage handling is exercised.
            (root / split / "images/example.jpg").write_bytes(split.encode())
            (root / split / "labels/example.txt").write_text(
                "\n".join(f"{i} 0.5 0.5 0.2 0.2" for i in range(9))
            )
        source = root / "data.yaml"
        source.write_text(yaml.safe_dump({"names": names, "train": "train/images",
                                         "val": "valid/images", "test": "test/images"}))
        return source, names

    def test_preserves_export_class_id_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, names = self.fixture(root)
            result = prepare_data(source, root / "prepared")
            self.assertEqual(yaml.safe_load(result.read_text())["names"], names)

    def test_rejects_leaked_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.fixture(root)
            (root / "valid/images/example.jpg").write_bytes(b"train")
            with self.assertRaisesRegex(ValueError, "Identical image"):
                prepare_data(source, root / "prepared")


if __name__ == "__main__":
    unittest.main()
