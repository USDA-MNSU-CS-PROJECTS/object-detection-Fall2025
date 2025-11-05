import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = "best_2_class_hpc.pt"
TEST_DIR = "test_images"
RAW_PRED_DIR = "predictions"
POST_DIR = "post_processed"
CSV_PATH = os.path.join(POST_DIR, "results.csv")
PIXEL_TO_MICRON = 0.9785316641067333  # This is a value embedded in the nds files, for 10x stitched images

os.makedirs(RAW_PRED_DIR, exist_ok=True)
os.makedirs(POST_DIR, exist_ok=True)

# === Load model ===
model = YOLO(MODEL_PATH)

# === Helper functions ===
def polygon_area(points):
    return Polygon(points).area

def polygon_center(points):
    poly = Polygon(points)
    return poly.centroid.x, poly.centroid.y

def find_main_cross_section(preds, img_shape):
    """Find cross-section closest to image center."""
    h, w = img_shape[:2]
    img_center = np.array([w / 2, h / 2])
    min_dist = float("inf")
    main_poly = None
    main_box = None

    for p in preds:
        if p["class_name"] == "Cross Section":
            cx, cy = polygon_center(p["points"])
            dist = np.linalg.norm(img_center - np.array([cx, cy]))
            if dist < min_dist:
                min_dist = dist
                main_poly = Polygon(p["points"])
                main_box = p
    return main_poly, main_box


def draw_research_style(img, main_poly, vb_polys, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Draw main cross-section outline
    if main_poly:
        x, y = main_poly.exterior.xy
        ax.plot(x, y, color="red", linewidth=1.5)
    # Draw filled vascular bundles
    for vb in vb_polys:
        vb_poly = Polygon(vb)
        x, y = vb_poly.exterior.xy
        ax.fill(x, y, color="green", alpha=0.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# === Inference + Post-processing ===
records = []

for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        continue

    img_path = os.path.join(TEST_DIR, img_name)
    results = model.predict(img_path, save=True, save_txt=False, project=RAW_PRED_DIR, name="run", exist_ok=True, verbose=False, conf=0.50)
    result = results[0]

    if result.masks is None:
        print(f"No masks detected in {img_name}")
        continue

    # Collect polygons
    preds = []
    for i, mask_polygon in enumerate(result.masks.xy):
        cls = int(result.boxes.cls[i])
        bbox = result.boxes[i]
        preds.append({
            "points": mask_polygon,
            "class_name": model.names[cls],
            "area": polygon_area(mask_polygon),
            "box": bbox.xyxy[0].cpu().numpy()
        })

    # Find main cross-section
    img = cv2.imread(img_path)
    main_poly, main_box = find_main_cross_section(preds, img.shape)

    if main_poly is None:
        print(f"No cross-section detected in {img_name}")
        continue

    # Filter vascular bundles inside main cross-section's bounding box
    main_cs_bbox = main_box['box']
    vb_polys = []
    for p in preds:
        if p["class_name"] == "Vascular Bundles":
            vb_center_x, vb_center_y = polygon_center(p["points"])
            if (vb_center_x >= main_cs_bbox[0] and vb_center_x <= main_cs_bbox[2] and
                vb_center_y >= main_cs_bbox[1] and vb_center_y <= main_cs_bbox[3]):
                vb_polys.append(p["points"])

    # Save post-processed visualization
    output_path = os.path.join(POST_DIR, img_name)
    draw_research_style(img, main_poly, vb_polys, output_path)

    # Record summary
    main_cx, main_cy = polygon_center(main_box["points"])
    vb_pixel_area = sum([polygon_area(vb) for vb in vb_polys])
    cs_pixel_area = main_box["area"]
    records.append({
        "image_name": img_name,
        # "main_cs_center_x": main_cx,
        # "main_cs_center_y": main_cy,
        "vb_count": len(vb_polys),
        # "vb_mask_area_pixels": vb_pixel_area,
        # "cs_mask_area_pixels": cs_pixel_area,
        "vb_area_microns": vb_pixel_area * (PIXEL_TO_MICRON**2),
        "cs_area_microns": cs_pixel_area * (PIXEL_TO_MICRON**2)
    })

# === Save CSV ===
pd.DataFrame(records).to_csv(CSV_PATH, index=False)
print(f"\n✅ Post-processing complete. Results saved in '{POST_DIR}' and '{CSV_PATH}'")
