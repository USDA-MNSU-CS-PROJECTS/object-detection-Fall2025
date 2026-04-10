import os
import sys
import shutil
import unittest
from unittest.mock import MagicMock

# Add the app directory to the path to allow imports from main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_full_pipeline

class TestPipeline(unittest.TestCase):

    def test_full_pipeline(self):
        """
        Tests the full analysis pipeline with a sample PNG image.
        """
        # Define the path to the sample image relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        sample_image_path = os.path.join(project_root, "src", "data_downloader", "initial_test_result.png")
        
        self.assertTrue(os.path.exists(sample_image_path), f"Sample image not found at {sample_image_path}")

        # Create a mock file object that mimics Gradio's File object
        mock_file = MagicMock()
        mock_file.name = sample_image_path
        
        # Run the pipeline
        csv_path, zip_path, status, df, temp_dir = run_full_pipeline(
            files=[mock_file],
            conf_threshold=0.25,
            save_overlays=False
        )
        
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
