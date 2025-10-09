from PIL import Image
import os

def convert_tiff_to_png(input_path: str, output_dir: str) -> str:
    """Converts a TIFF file to a PNG file.

    Args:
        input_path: The absolute path to the input TIFF file.
        output_dir: The absolute path to the directory where the PNG file will be saved.

    Returns:
        The absolute path to the converted PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(output_dir, f"{base_name}.png")

    # Open input image
    try:
        input_image = Image.open(input_path).convert("RGB")  # convert to RGB
    except FileNotFoundError:
        print(f"Error: Input image '{input_path}' not found.")
        raise

    # Convert to PNG
    print(f"Converting to PNG: {os.path.basename(input_path)} ...")

    # Save result
    input_image.save(output_path, "PNG")

    print(f"Conversion successful. Output saved to: {output_path}")
    
    return output_path
