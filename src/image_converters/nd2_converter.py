import nd2
import tifffile
import numpy as np
import os

def convert_nd2_to_tiff(input_path: str, output_dir: str) -> str:
    """Converts an ND2 file to a TIFF file.

    Args:
        input_path: The absolute path to the input ND2 file.
        output_dir: The absolute path to the directory where the TIFF file will be saved.

    Returns:
        The absolute path to the converted TIFF file.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(output_dir, f"{base_name}.tiff")

    # ---- LOAD ND2 IMAGE ----
    try:
        data = nd2.imread(input_path)
        print(f"Original ND2 shape: {data.shape}, dtype: {data.dtype}")
    except FileNotFoundError:
        print(f"Error: Input image '{input_path}' not found.")
        raise

    # ---- CONVERT TO 8-BIT FOR VIEWING ----
    # Scale all pixel values to 0-255
    data_8bit = (data / data.max() * 255).astype(np.uint8)

    # ---- HANDLE CHANNEL ORDER (Optional) ----
    # If the image has 3 channels, ensure it's RGB
    if data_8bit.ndim == 4:  # shape: (frames, height, width, channels)
        data_8bit = data_8bit[0]  # take first frame if multiple frames

    if data_8bit.shape[-1] == 3:
        # ND2 sometimes stores as BGR → convert to RGB
        data_8bit = data_8bit[..., ::-1]

    # ---- SAVE TIFF ----
    tifffile.imwrite(output_path, data_8bit, compression=None)
    print(f"8-bit TIFF saved at {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path
