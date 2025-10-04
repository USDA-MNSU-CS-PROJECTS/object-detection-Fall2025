from PIL import Image
import os

def main() -> None:
    # Resolve repo-relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # .../src
    data_dir = os.path.join(project_root, "data")
    output_root = os.path.join(data_dir, "main_images", "output_images")
    tiff_dir = os.path.join(output_root, "tiff_images")
    png_dir = os.path.join(output_root, "png_images")

    # Ensure output directories exist
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(tiff_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Collect all TIFFs to PNGs
    tiff_candidates = [f for f in os.listdir(tiff_dir) if f.lower().endswith((".tiff", ".tif"))]
    if not tiff_candidates:
        raise FileNotFoundError(
            f"No TIFF files found in '{tiff_dir}'. Run tiff_converter first or add a TIFF."
        )

    for filename in tiff_candidates:
        input_path = os.path.join(tiff_dir, filename)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(png_dir, f"{base_name}.png")

        # Open input image
        try:
            input_image = Image.open(input_path).convert("RGB")  # convert to RGB
        except FileNotFoundError:
            print(f"Error: Input image '{input_path}' not found.")
            raise

        # Convert to PNG
        print(f"Converting to PNG: {filename} ...")

        # Save result
        input_image.save(output_path, "PNG")

        print(f"Conversion successful. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
