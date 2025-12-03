"""
Setup script to create necessary data directory structure for images.
Run this after cloning the repository.
"""
import os

def create_directory_structure():
    """Create all necessary image data directories"""

    # Define the directory structure
    directories = [
        "data/nd2_images/input_images",
        "data/main_images/output_images/tiff_images",
        "data/main_images/output_images/png_images",
        "converter_test_data/multi_nd2_files",
        "converter_test_data/output_pngs"
    ]

    project_root = os.path.dirname(os.path.abspath(__file__))

    print("Creating data directory structure...")

    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your .nd2 files in: data/nd2_images/input_images/")
    print("2. Run the converter scripts to process images")

if __name__ == "__main__":
    create_directory_structure()
