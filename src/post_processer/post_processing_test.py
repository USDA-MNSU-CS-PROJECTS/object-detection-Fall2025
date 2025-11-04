import pandas as pd
import os
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

PIXEL_TO_MICRON = 0.9785316641067333  # Example conversion factor, adjust as needed

def polygon_area(points):
    return Polygon(points).area

def polygon_center(points):
    poly = Polygon(points)
    return poly.centroid.x, poly.centroid.y

def calculate_areas(vb_pixel_area, cs_pixel_area):
    """
    Calculate the areas in microns for VB and CS masks.

    Args:
        vb_pixel_area (float): Area of VB mask in pixels.
        cs_pixel_area (float): Area of CS mask in pixels.

    Returns:
        dict: A dictionary containing the areas in microns.
    """
    return {
        "vb_area_microns": vb_pixel_area * (PIXEL_TO_MICRON**2),
        "cs_area_microns": cs_pixel_area * (PIXEL_TO_MICRON**2)
    }

def generate_records(image_name, vb_polys, vb_pixel_area, cs_pixel_area):
    """
    Generate a record for a single image's post-processing results.

    Args:
        image_name (str): Name of the image.
        vb_polys (list): List of VB polygons.
        vb_pixel_area (float): Area of VB mask in pixels.
        cs_pixel_area (float): Area of CS mask in pixels.

    Returns:
        dict: A dictionary containing the processed record.
    """
    areas = calculate_areas(vb_pixel_area, cs_pixel_area)
    return {
        "image_name": image_name,
        "vb_count": len(vb_polys),
        "vb_area_microns": areas["vb_area_microns"],
        "cs_area_microns": areas["cs_area_microns"]
    }

def save_to_csv(records, csv_path):
    """
    Save the processed records to a CSV file.

    Args:
        records (list): List of dictionaries containing processed data.
        csv_path (str): Path to save the CSV file.
    """
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\n✅ Results saved in '{csv_path}'")
    return

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

def draw_research_style(img, main_poly, vb_polys, output_path, save_image=True):
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
    if not save_image:
        return fig, ax
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return

# use the below style to get the result of the model
# results = model.predict(img_path, save=True, save_txt=False, project=RAW_PRED_DIR, name="run", exist_ok=True, verbose=False, conf=0.50)
# result = results[0]

def collect_polygons(result, model):
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
    return preds

# Example Find main cross-section
# img = cv2.imread(img_path)
# main_poly, main_box = find_main_cross_section(preds, img.shape)

def filter_vascular_bundles(main_poly, main_box, preds, img_name):
    if main_poly is None:
        print(f"No cross-section detected in {img_name}")
        

    # Filter vascular bundles inside main cross-section's bounding box
    main_cs_bbox = main_box['box']
    vb_polys = []
    for p in preds:
        if p["class_name"] == "Vascular Bundles":
            vb_center_x, vb_center_y = polygon_center(p["points"])
            if (vb_center_x >= main_cs_bbox[0] and vb_center_x <= main_cs_bbox[2] and
                vb_center_y >= main_cs_bbox[1] and vb_center_y <= main_cs_bbox[3]):
                vb_polys.append(p["points"])
    return vb_polys

def save_viz(POST_DIR, img, main_poly, vb_polys, img_name):
    # Save post-processed visualization
    output_path = os.path.join(POST_DIR, img_name)
    draw_research_style(img, main_poly, vb_polys, output_path)
    


# Example Find main cross-section
# img = cv2.imread(img_path)
# main_poly, main_box = find_main_cross_section(preds, img.shape)
