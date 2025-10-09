import os
import dotenv
from boxsdk import OAuth2, Client
import sys
import zipfile
import shutil

# Add image_converters to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../image_converters')))

from nd2_converter import convert_nd2_to_tiff
from png_converter import convert_tiff_to_png

# Define directories
base_output_dir = os.path.abspath('../data')
zip_output_dir = os.path.join(base_output_dir, 'zip_files')

os.makedirs(zip_output_dir, exist_ok=True)

# Download files
total_count = 0
total_t0_and_t4_count = 0

# Load from .env file
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '.env'), override=True)
CLIENT_ID = os.getenv('BOX_CLIENT_ID')
CLIENT_SECRET = os.getenv('BOX_CLIENT_SECRET')
REFRESH_TOKEN = os.getenv('BOX_REFRESH_TOKEN')

# Initialize with auto-refresh
oauth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    refresh_token=REFRESH_TOKEN
)

client = Client(oauth)

folder_id = '340825252478'  # 10x_stitched_images
# folder_id = '340827995628' # sub-folder for testing
folder = client.folder(folder_id)

for batch_folder in folder.get_items():
    if batch_folder.type == 'folder':
        print(f"Entering sub-folder: {batch_folder.name}")
        batch_folder_obj = client.folder(batch_folder.id)
        for pg_or_rr_folder in batch_folder_obj.get_items():
            if pg_or_rr_folder.type == 'folder' and ('pg' in pg_or_rr_folder.name.lower()): # selecting only pg folders for now
                print(f"  Entering sub-sub-folder: {pg_or_rr_folder.name}")
                pg_folder_obj = client.folder(pg_or_rr_folder.id)
                
                # Create a directory for the sub-folder
                sub_folder_dir = os.path.join(base_output_dir, pg_or_rr_folder.name)
                os.makedirs(sub_folder_dir, exist_ok=True)

                folder_count = 0
                folder_t0_and_t4_count = 0
                for item in pg_folder_obj.get_items():
                    if item.type == 'file' and item.name.lower().endswith('.nd2'):
                        folder_count += 1
                        total_count += 1
                        # Optional: Filter for DL data (e.g., images)
                        if 't0_' in item.name.lower() or 't4_' in item.name.lower():
                            total_t0_and_t4_count += 1
                            folder_t0_and_t4_count += 1
                            
                            nd2_file_path = os.path.join(sub_folder_dir, item.name)
                            with open(nd2_file_path, 'wb') as f:
                                item.download_to(f)

                            try:
                                # Convert ND2 to TIFF
                                tiff_file_path = convert_nd2_to_tiff(nd2_file_path, sub_folder_dir)

                                # Convert TIFF to PNG
                                png_file_path = convert_tiff_to_png(tiff_file_path, sub_folder_dir)

                                print(f'    Successfully converted {item.name} to {png_file_path}')

                                # Clean up intermediate files
                                os.remove(nd2_file_path)
                                os.remove(tiff_file_path)

                            except Exception as e:
                                print(f'Error converting {item.name}: {e}')

                print(f"  Folder '{pg_or_rr_folder.name}' has {folder_count} files, {folder_t0_and_t4_count} T0 or T4 files.")
                
                # Zip the directory
                zip_file_path = os.path.join(zip_output_dir, f'{pg_or_rr_folder.name}.zip')
                with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(sub_folder_dir):
                        for file in files:
                            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), sub_folder_dir))
                
                # Remove the directory
                shutil.rmtree(sub_folder_dir)
                
                print("  -----------------------------------")

# Verify
print(f"Total files in folder: {total_count}")
print(f"Files matching criteria (T0/T4): {total_t0_and_t4_count}")
print("Test complete. Files ready for annotation/training.")
