import os
import sys
import shutil
import unittest
from unittest.mock import MagicMock

_app = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_third = os.path.abspath(os.path.join(_app, "..", "third_party"))
for _p in (_third, _app):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_app, "api"))

from config.inference_constants import MODEL_A_CASPARIAN_EPIDERMIS, MODEL_B_VASCULAR_BUNDLES
from main import run_full_pipeline


class TestPipeline(unittest.TestCase):

    def test_full_pipeline(self):
        """
        Tests the full analysis pipeline with a sample PNG image.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        sample_image_path = os.path.join(project_root, "src", "data_downloader", "initial_test_result.png")
        path_a = os.path.join(project_root, "sample_trained_models", MODEL_A_CASPARIAN_EPIDERMIS)
        path_b = os.path.join(project_root, "sample_trained_models", MODEL_B_VASCULAR_BUNDLES)

        self.assertTrue(os.path.exists(sample_image_path), f"Sample image not found at {sample_image_path}")
        if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
            self.skipTest(
                f"Skipping pipeline test: place YOLO weights as {MODEL_A_CASPARIAN_EPIDERMIS} and "
                f"{MODEL_B_VASCULAR_BUNDLES} in sample_trained_models/"
            )

        mock_file = MagicMock()
        mock_file.name = sample_image_path

        csv_path, zip_path, status, df, temp_dir = run_full_pipeline(files=[mock_file])
        
        try:
            # --- Assertions ---
            self.assertIsNotNone(temp_dir, "Temporary directory should have been created")
            self.assertTrue(os.path.isdir(temp_dir), "Temporary directory should exist")
            
            if not df.empty:
                self.assertIsNotNone(csv_path, "CSV path should be generated if detections are found")
                self.assertTrue(os.path.exists(csv_path), "CSV file should exist if detections are found")
            else:
                self.assertIsNone(csv_path, "CSV path should be None if no detections are found")
                print("\nNote: No detections were found in the sample image, which is acceptable.")
            
            self.assertIsNotNone(zip_path, "ZIP path should not be None")
            self.assertTrue(os.path.exists(zip_path), "Output ZIP file should exist")
            
            self.assertNotIn("Error", status, f"Pipeline status should not contain 'Error'. Status: {status}")
            self.assertIsNotNone(df, "Result DataFrame should not be None")
            
            print("\nPipeline test completed successfully.")
            print(f"Status: {status}")
            
        finally:
            # --- Cleanup ---
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    # This allows the test to be run directly
    unittest.main()
