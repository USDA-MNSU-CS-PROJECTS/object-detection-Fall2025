"""
Simple ND2 to TIFF Converter
Just converts ND2 files to TIFF - easy to use!
"""

import os
from nd2reader import ND2Reader
from tifffile import imwrite

def convert_single_file(nd2_file, verbose=True):
    """Convert one ND2 file to TIFF"""
    # If verbose is true, the update messages will display in terminal
    # If false, no messages will appear, useful for pipeline automation
    if verbose:
        print(f"Converting: {nd2_file}")
    
    # Open the ND2 file
    with ND2Reader(nd2_file) as images:
        if verbose:
            print(f"  Found {len(images)} layers/images")
            
		# Get base filename without extension
        base_name = nd2_file.replace('.nd2', '')
        
        output_files = []
		
        # Get the first (or only) image
        if len(images) == 1:
            image_data = images[0]
            output_file = f"{base_name}.tiff"
            imwrite(output_file, image_data)
            output_files.append(output_file)
            if verbose:
                print(f"  Saved: {output_file}")
        else:
            # If multiple images, save each layer separately
            for i, img in enumerate(images):
                output_file = f"{base_name}_layer_{i+1:02d}.tiff"
                imwrite(output_file, img)
                output_files.append(output_file)
                if verbose:
                    print(f"  Saved layer {i+1}: {output_file}")
    
    return output_files

def convert_folder(folder_path, verbose=True):
    """Convert all ND2 files in a folder"""
    if verbose:
        print(f"Looking for ND2 files in: {folder_path}")
    
    # Find all ND2 files in the folder
    nd2_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.nd2'):
            full_path = os.path.join(folder_path, file)
            nd2_files.append(full_path)
    
    if verbose:
        print(f"Found {len(nd2_files)} ND2 files")
    
    # Convert each file
    all_output_files = []
    total_layers = 0
    
    for nd2_file in nd2_files:
        try:
            output_files = convert_single_file(nd2_file, verbose=verbose)
            all_output_files.extend(output_files)
            total_layers += len(output_files)
        except Exception as e:
            if verbose:
                print(f"Error with {nd2_file}: {e}")
    
    if verbose:
        print(f"Successfully converted {len(nd2_files)} ND2 files into {total_layers} TIFF files")
    return all_output_files

# Simple usage examples
if __name__ == "__main__":

    print("ND2 to TIFF Converter")
    print("1. Convert a single file")
    print("2. Convert all files in a folder")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        file_path = input("Enter the path to your ND2 file: ")
        if os.path.exists(file_path):
            convert_single_file(file_path)
        else:
            print("File not found!")
    elif choice == "2":
        folder_path = input("Enter the path to your folder: ")
        if os.path.exists(folder_path):
            convert_folder(folder_path)
        else:
            print("Folder not found!")
    else:
        print("Invalid choice!")
