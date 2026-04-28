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

YOLO_CLASS_CASPARIAN = 1
YOLO_CLASS_EPIDERMIS = 0

# If True: model A labels are inverted vs biology (outer contour named "Casparian Strip",
# inner ring named "Epidermis"). We map to bucket 0=casparian 1=epidermis for metrics/ROI.
MODEL_A_SWAP_EPI_CASP_LABELS = True

# If True: when multiple epidermis masks exist, keep only the one whose centroid is nearest
# image center (drops stray fragments e.g. in corners). Tie-break: larger polygon area.
MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER = True

DEFAULT_CONF_MODEL_A = 0.8
DEFAULT_CONF_MODEL_B = 0.5

PIXEL_TO_MICRON = 0.9785316641067333

# Post-processor output toggles (dual stem pipeline / output_images.zip).
# Overlay PNGs: output_dir/visualizations/. Per-metric debug PNGs: output_dir/metric_debug_viz/.
PIPELINE_WRITE_OVERLAY_VISUALIZATIONS = True
PIPELINE_EXPORT_METRIC_DEBUG_VIZ = False

# Ring mask quality for Mean_epi_casp_distance (medial-axis thickness):
# degraded if largest component has less than this fraction of total ring pixels.
RING_QUALITY_MIN_MAIN_COMPONENT_AREA_FRACTION = 0.75
# warn if morphological closing increases ring area by more than this factor (gaps/narrow breaks).
RING_QUALITY_CLOSURE_AREA_WARN_RATIO = 1.12
# iterations of binary closing for the closure test.
RING_QUALITY_CLOSURE_ITERATIONS = 2
# warn if pruned medial axis has more than this many branch pixels (3+ neighbors on skeleton).
RING_QUALITY_MEDIAL_BRANCH_WARN = 40

# Before medial axis: binary_opening iterations on ring (0 = off). Reduces boundary spurs.
RING_MASK_OPEN_BEFORE_MEDIAL_ITERATIONS = 3
# After medial_axis: remove 8-connected leaf pixels up to this many passes (spur pruning).
RING_MEDIAL_SPUR_PRUNE_MAX_ITER = 24

