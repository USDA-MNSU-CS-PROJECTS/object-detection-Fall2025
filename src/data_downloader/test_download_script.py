import os
import dotenv
from boxsdk import OAuth2, Client

# Load from .env file
dotenv.load_dotenv('.env', override=True)
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

# Download to this dir
output_dir = '../data'
os.makedirs(output_dir, exist_ok=True)

# Download files
folder = client.folder(folder_id)
# tst_download = True # For testing, download only one file
total_count = 0
total_t0_and_t4_count = 0
for batch_folder in folder.get_items():
    if batch_folder.type == 'folder':
        print(f"Entering sub-folder: {batch_folder.name}")
        batch_folder_obj = client.folder(batch_folder.id)
        for pg_or_rr_folder in batch_folder_obj.get_items():
            if pg_or_rr_folder.type == 'folder' and ('pg' in pg_or_rr_folder.name.lower()): # selecting only pg folders for now
                print(f"  Entering sub-sub-folder: {pg_or_rr_folder.name}")
                pg_folder_obj = client.folder(pg_or_rr_folder.id)
                folder_count = 0
                folder_t0_and_t4_count = 0
                for item in pg_folder_obj.get_items():
                    if item.type == 'file':
                        folder_count += 1
                        total_count += 1
                        # Optional: Filter for DL data (e.g., images)
                        if 't0_' in item.name.lower() or 't4_' in item.name.lower():
                            total_t0_and_t4_count += 1
                            folder_t0_and_t4_count += 1
                            # if tst_download:
                            #     file_path = os.path.join(output_dir, item.name)
                            #     with open(file_path, 'wb') as f:
                            #         item.download_to(f)
                            #     print(f"    Downloaded: {file_path}")
                            #     tst_download = False
                            # print(f"    Found file: {item.name}, skipping download for test.")
                print(f"  Folder '{pg_or_rr_folder.name}' has {folder_count} files, {folder_t0_and_t4_count} T0 or T4 files.")
                print("  -----------------------------------")
# print(f"Total so far: {total_count} files, {total_t0_and_t4_count} T0 or T4 files.")
# Code for batch_folder testing
"""
for item in folder.get_items():
    if item.type == 'file':
        total_count += 1
        # Optional: Filter for DL data (e.g., images)
        if 't0_' in item.name.lower() or 't4_' in item.name.lower():
            t0_and_t4_count += 1
            if tst_download:
                file_path = os.path.join(output_dir, item.name)
                with open(file_path, 'wb') as f:
                    item.download_to(f)
                print(f"Downloaded: {file_path}")
                tst_download = False
            print(f"Found file: {item.name}, skipping download for test.")
"""

# Verify
print(f"Total files in folder: {total_count}")
print(f"Files matching criteria (T0/T4): {total_t0_and_t4_count}")
print("Test complete. Files ready for annotation/training.")