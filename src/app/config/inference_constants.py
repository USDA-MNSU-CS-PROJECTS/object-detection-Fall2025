"""
Model weight filenames and YOLO class name strings.
Edit class names to match your models' data.yaml when weights are ready.
"""

# Expected under sample_trained_models/ (add .pt files locally; not committed)
MODEL_A_CASPARIAN_EPIDERMIS = "casparian_epidermis.pt"
MODEL_B_VASCULAR_BUNDLES = "vascular_bundles.pt"

CLASS_CASPARIAN = "Casparian Strip"
CLASS_EPIDERMIS = "Epidermis"
CLASS_CASPARIAN_FALLBACK = "Cross Section"

CLASS_VASCULAR_BUNDLES = "Vascular Bundles"

YOLO_CLASS_CASPARIAN = 0
YOLO_CLASS_EPIDERMIS = 1

# If True: model A labels are inverted vs biology (outer contour named "Casparian Strip",
# inner ring named "Epidermis"). We map to bucket 0=casparian 1=epidermis for metrics/ROI.
MODEL_A_SWAP_EPI_CASP_LABELS = True

# If True: when multiple epidermis masks exist, keep only the one whose centroid is nearest
# image center (drops stray fragments e.g. in corners). Tie-break: larger polygon area.
MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER = True

DEFAULT_CONF_MODEL_A = 0.5
DEFAULT_CONF_MODEL_B = 0.8

PIXEL_TO_MICRON = 0.9785316641067333
