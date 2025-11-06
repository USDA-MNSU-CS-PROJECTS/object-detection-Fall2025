import os
import numpy as np
from PIL import Image
from typing import List, Optional

# Assuming ImageConverter is in the same directory or importable from a known path
# If converter.py is in src/app/api/, and this test file is in the same directory,
# then the import would be:
from converter import ImageConverter

# If this test file is elsewhere, adjust the import path accordingly.
# For example, if test_converter.py is in the project root:
# from src.app.api.converter import ImageConverter


if __name__ == "__main__":
    converter = ImageConverter()

    # --- Setup paths ---
    # Define a temporary directory for test files
    test_data_root = "converter_test_data"
    input_single_file_path = os.path.join(test_data_root, "sample_single.nd2")
    input_multi_file_dir = os.path.join(test_data_root, "multi_nd2_files")
    output_dir = os.path.join(test_data_root, "output_pngs")

    # Create test directories
    os.makedirs(input_multi_file_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Test Setup ---")
    print(f"Please ensure you have actual .nd2 files for testing:")
    print(f"1. For single file conversion: Place an .nd2 file at: {input_single_file_path}")
    print(f"2. For directory conversion: Place .nd2 files inside the directory: {input_multi_file_dir}")
    print(f"   (e.g., {os.path.join(input_multi_file_dir, 'image1.nd2')}, {os.path.join(input_multi_file_dir, 'image2.nd2')})")
    print(f"Output PNGs will be saved to: {output_dir}\n")

    input("Press Enter to continue after placing your .nd2 test files...")

    # --- Test single file conversion ---
    print("\n--- Testing single ND2 file conversion ---")
    if os.path.exists(input_single_file_path):
        converted_path = converter.convert_nd2_to_png(input_single_file_path, output_dir)
        if converted_path:
            print(f"Single file converted successfully: {converted_path}")
            if os.path.exists(converted_path):
                print(f"Output PNG file exists at: {converted_path}")
            else:
                print(f"Error: Output PNG file not found at: {converted_path}")
        else:
            print(f"Single file conversion failed for {input_single_file_path}")
    else:
        print(f"Skipping single file test: {input_single_file_path} not found.")

    # --- Test directory conversion ---
    print("\n--- Testing directory conversion ---")
    nd2_files_in_dir = [f for f in os.listdir(input_multi_file_dir) if f.lower().endswith(".nd2")]
    if nd2_files_in_dir:
        print(f"Found {len(nd2_files_in_dir)} .nd2 files in {input_multi_file_dir}. Proceeding with directory conversion.")
        converted_paths_list = converter.process_directory(input_multi_file_dir, output_dir)
        if converted_paths_list:
            print(f"Directory conversion successful. Converted {len(converted_paths_list)} files:")
            for path in converted_paths_list:
                print(f"  - {path}")
                if not os.path.exists(path):
                    print(f"Error: Output PNG file not found at: {path}")
        else:
            print(f"Directory conversion failed for files in {input_multi_file_dir}")
    else:
        print(f"Skipping directory test: No .nd2 files found in {input_multi_file_dir}.")

    print(f"\nTest complete. Check the '{output_dir}' directory for converted PNGs.")
    print(f"You can clean up the test data by deleting the '{test_data_root}' directory.")
