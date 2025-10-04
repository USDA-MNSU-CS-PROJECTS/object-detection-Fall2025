from .tiff_converter import main as convert_tiff
from .png_converter import main as convert_png

__all__ = ['convert_tiff', 'convert_png']

# Usage - when importing the above methods into another file use:
# from scr.image_converters import convert_tiff, convert_png