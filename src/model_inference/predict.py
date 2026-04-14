import os
import sys
import json
from ultralytics import YOLO

# Paths relative to project root (predict.py is in src/model_inference/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_APP_DIR = os.path.join(_PROJECT_ROOT, "src", "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from config.inference_constants import DEFAULT_CONF_MODEL_B, MODEL_B_VASCULAR_BUNDLES

MODEL_PATH = os.path.join(_PROJECT_ROOT, "sample_trained_models", MODEL_B_VASCULAR_BUNDLES)
TEST_IMAGES = os.path.join(_PROJECT_ROOT, "test_images")  # folder or single image path
DEBUG_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "debug_output")


def _save_raw_predictions_debug(predictions, output_dir, source_label="predict"):
    """Save raw model output (counts, centers, bbox, conf, area) for debugging and comparison with main."""
    try:
        from shapely.geometry import Polygon
        records = []
        for result in predictions:
            img_name = os.path.basename(result.path)
            entry = {
                "image_name": img_name,
                "source": source_label,
                "counts_by_class": {},
                "vascular_bundles": [],
                "cross_sections": [],
            }
            if result.masks is None:
                entry["counts_by_class"] = {"Vascular Bundles": 0, "Cross Section": 0}
                records.append(entry)
                continue
            for i, mask_xy in enumerate(result.masks.xy):
                cls_idx = int(result.boxes.cls[i])
                class_name = result.names[cls_idx]
                entry["counts_by_class"][class_name] = entry["counts_by_class"].get(class_name, 0) + 1
                bbox = result.boxes[i].xyxy[0].cpu().numpy().tolist()
                conf = float(result.boxes.conf[i].cpu().numpy()) if result.boxes.conf is not None else None
                try:
                    poly = Polygon(mask_xy)
                    cx, cy = float(poly.centroid.x), float(poly.centroid.y)
                    area_px = float(poly.area)
                    perimeter_px = float(poly.length)
                except Exception:
                    cx, cy, area_px, perimeter_px = None, None, None, None
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                area_bbox = (x2 - x1) * (y2 - y1) if len(bbox) >= 4 else None
                item = {
                    "center_xy": [cx, cy],
                    "bbox_xyxy": bbox,
                    "confidence": conf,
                    "area_pixels": area_px,
                    "perimeter_pixels": perimeter_px,
                    "area_bbox": area_bbox,
                    "n_contour_points": len(mask_xy),
                }
                if class_name == "Vascular Bundles":
                    entry["vascular_bundles"].append(item)
                elif class_name == "Cross Section":
                    entry["cross_sections"].append(item)
            records.append(entry)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "raw_predictions_debug_from_predict.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Debug: raw predictions saved to {path}")
    except Exception as e:
        print(f"Debug save failed: {e}")


# Load the trained segmentation model
model = YOLO(MODEL_PATH)

# Run prediction on your local images
results = model.predict(
    source=TEST_IMAGES,
    save=True,
    imgsz=640,
    conf=DEFAULT_CONF_MODEL_B,
)

# Save full raw output to project debug_output for comparison with main pipeline
_save_raw_predictions_debug(results, DEBUG_OUTPUT_DIR, source_label="predict")
