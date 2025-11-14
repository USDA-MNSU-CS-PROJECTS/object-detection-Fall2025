import os
import nd2
import numpy as np
from PIL import Image
from typing import List, Optional

class ImageConverter:
    def __init__(self):
        pass

    def convert_nd2_to_png(self, input_path: str, output_dir: str) -> Optional[str]:
        """
        Converts a single ND2 file to PNG, preserving channel orientation
        and avoiding intermediate TIFF storage.

        Args:
            input_path: Path to ND2 file.
            output_dir: Directory to save PNG.

        Returns:
            Full path to the saved PNG file, or None if conversion fails.
        """
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")

        try:
            # Load ND2 → numpy array
            img = nd2.imread(input_path)
            img = np.asarray(img)

            # Handle multi-frame ND2 by taking the first frame
            if img.ndim > 3:
                img = img[0]

            # If channel-first (C, H, W) → convert to (H, W, C)
            if img.ndim == 3:
                if img.shape[0] in [3, 4] and img.shape[-1] not in [3, 4]:
                    img = np.transpose(img, (1, 2, 0))
            else:
                img = np.expand_dims(img, axis=-1)

            # Convert to 8-bit
            if np.max(img) > 0:
                img_8bit = (img / np.max(img) * 255).astype(np.uint8)
            else:
                img_8bit = img.astype(np.uint8)

            # If grayscale, repeat channels
            if img_8bit.shape[-1] == 1:
                img_8bit = np.repeat(img_8bit, 3, axis=-1)

            # Ensure shape is H x W x 3
            if img_8bit.shape[-1] > 3:
                img_8bit = img_8bit[..., :3]

            # If the image has 3 channels, assume it's BGR and convert to RGB for PIL
            if img_8bit.shape[-1] == 3:
                img_8bit = img_8bit[..., ::-1]

            # Save final PNG
            Image.fromarray(img_8bit).save(output_path, format="PNG")
            print(f"Converted {input_path} → {output_path}")

            return output_path
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_path}")
            return None
        except Exception as e:
            print(f"An error occurred while converting {input_path}: {e}")
            return None

    def process_directory(self, input_dir: str, output_dir: str) -> List[str]:
        """
        Converts all ND2 files in a directory to PNG. Ignores non-ND2 files.

        Args:
            input_dir: Folder with .nd2 files
            output_dir: Folder where .png files will be stored

        Returns:
            List of full paths to PNG files generated.
        """
        os.makedirs(output_dir, exist_ok=True)

        nd2_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(".nd2") and not f.startswith('._')
        ]

        if len(nd2_files) == 0:
            print("⚠️ No ND2 files found in input directory.")
            return []

        output_paths = []

        for filename in nd2_files:
            full_input = os.path.join(input_dir, filename)
            png_path = self.convert_nd2_to_png(full_input, output_dir)
            if png_path:
                output_paths.append(png_path)

        print(f"✅ Converted {len(output_paths)} ND2 files to PNG.")
        return output_paths
