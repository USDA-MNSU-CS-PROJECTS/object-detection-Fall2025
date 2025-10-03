from rembg import remove
from PIL import Image
import os

def main() -> None:
    # Resolve repo-relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # .../src
    data_dir = os.path.join(project_root, "data")
    output_root = os.path.join(data_dir, "main_images", "output_images")
    tiff_dir = os.path.join(output_root, "tiff_images")
    #jpg_dir = os.path.join(output_root, "jpg_images")
    preprocessed_dir = os.path.join(output_root, "preprocessed_images")

    # Ensure output directories exist
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(tiff_dir, exist_ok=True)
    #os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Collect all TIFFs to preprocess
    tiff_candidates = [f for f in os.listdir(tiff_dir) if f.lower().endswith((".tiff", ".tif"))]
    if not tiff_candidates:
        raise FileNotFoundError(
            f"No TIFF files found in '{tiff_dir}'. Run tiff_converter first or add a TIFF."
        )

    for filename in tiff_candidates:
        input_path = os.path.join(tiff_dir, filename)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(preprocessed_dir, f"{base_name}_removed.png")

        # Open input image
        try:
            input_image = Image.open(input_path).convert("RGB")  # convert to RGB
        except FileNotFoundError:
            print(f"Error: Input image '{input_path}' not found.")
            raise

        # Remove background
        print(f"Removing background for: {filename} ...")
        output_image = remove(input_image)

        # Save result
        output_image.save(output_path)

        print(f"Background removed successfully. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
