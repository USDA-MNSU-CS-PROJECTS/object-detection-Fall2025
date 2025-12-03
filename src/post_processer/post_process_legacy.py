# Example post-processing: pick main cross-section and filter vascular bundles
import numpy as np
from shapely.geometry import Point, Polygon
from skimage.measure import label, regionprops

def mask_to_polygons(mask):
    # crude conversion: get largest connected region polygon(s) via regionprops contours
    # For robust vectorization use cv2.findContours or skimage.measure.find_contours
    from skimage import measure
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    polygons = []
    for c in contours:
        # contour coords are (row, col) -> convert to (x, y)
        poly = [(float(x), float(y)) for x, y in zip(c[:,1], c[:,0])]
        if len(poly) >= 3:
            polygons.append(Polygon(poly))
    return polygons

def centroid_of_mask(mask):
    props = regionprops(label(mask.astype(np.uint8)))
    if not props:
        return None
    c = props[0].centroid  # (row, col)
    return (c[1], c[0])     # return (x, y)

def select_main_cross_section(cross_masks, img_shape):
    # cross_masks: list of boolean arrays for predicted cross_sections
    # pick based on proximity to image center + area:
    h, w = img_shape
    img_center = np.array([w/2.0, h/2.0])
    best_score = None
    best_idx = None
    for i, m in enumerate(cross_masks):
        props = regionprops(label(m.astype(np.uint8)))
        if not props:
            continue
        area = props[0].area
        centroid = np.array([props[0].centroid[1], props[0].centroid[0]])  # x,y
        dist = np.linalg.norm(centroid - img_center)
        # score: smaller distance better, larger area better
        score = (dist) - 0.001 * area   # tune weights; lower is better
        if best_score is None or score < best_score:
            best_score = score
            best_idx = i
    return best_idx

def filter_vbs_by_cross(main_cross_mask, vb_masks, vb_scores=None, iou_threshold=0.0):
    kept = []
    main_mask_bool = main_cross_mask.astype(bool)
    main_area = main_mask_bool.sum()
    for i, m in enumerate(vb_masks):
        # centroid containment
        cent = centroid_of_mask(m)
        in_main = False
        if cent is not None:
            x, y = int(round(cent[0])), int(round(cent[1]))
            if 0 <= y < main_mask_bool.shape[0] and 0 <= x < main_mask_bool.shape[1]:
                in_main = bool(main_mask_bool[y, x])
        if in_main:
            kept.append(i)
            continue
        # fallback: check mask IoU if centroid not in main
        if iou_threshold > 0.0:
            intersection = np.logical_and(main_mask_bool, m).sum()
            union = np.logical_or(main_mask_bool, m).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                kept.append(i)
    return kept

# Example usage:
# model_out = ... obtain from YOLOv8 results
# cross_masks = [mask for mask, cls in zip(model_out.masks, model_out.class_ids) if cls == CROSS_CLASS]
# vb_masks = [mask for mask, cls in zip(model_out.masks, model_out.class_ids) if cls == VB_CLASS]
# main_idx = select_main_cross_section(cross_masks, img_shape=(H,W))
# main_mask = cross_masks[main_idx]
# kept_vb_indices = filter_vbs_by_cross(main_mask, vb_masks, iou_threshold=0.05)
