import pandas as pd
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import matplotlib.pyplot as plt

class PostProcessor:
    def __init__(self, output_dir, pixel_to_micron_ratio=0.9785316641067333):
        """
        Initializes the PostProcessor.

        Args:
            output_dir (str): The directory to save post-processed visualizations.
            pixel_to_micron_ratio (float): The conversion factor from pixels to microns.
        """
        self.viz_output_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_output_dir, exist_ok=True)
        self.PIXEL_TO_MICRON = pixel_to_micron_ratio
        self.visualization_paths = []

    def process_predictions(self, prediction_results: list) -> pd.DataFrame:
        """
        Processes a list of prediction results from a YOLO model.

        Args:
            prediction_results (list): A list of ultralytics result objects.

        Returns:
            pd.DataFrame: A DataFrame containing the post-processed data.
        """
        records = []
        self.visualization_paths = []

        for result in prediction_results:
            img_path = result.path
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)

            if result.masks is None:
                print(f"Warning: No masks detected in {img_name}")
                continue

            preds = self._collect_polygons(result)
            main_poly, main_box = self._find_main_cross_section(preds, img.shape)

            if main_poly is None:
                print(f"Warning: No 'Cross Section' detected in {img_name}")
                records.append({
                    "image_name": img_name,
                    "vb_count": "N/A",
                    "vb_area_microns": "N/A",
                    "cs_area_microns": "N/A",
                    "vb_to_cs_ratio": "N/A",
                    "notes": "No Cross Section detected"
                })
                continue

            vb_polys = self._filter_vascular_bundles(main_box, preds)
            
            # Create complete cross section polygon that includes vascular bundles
            complete_cs_poly = self._create_complete_cross_section(main_poly, vb_polys)

            # Save visualization
            output_path = os.path.join(self.viz_output_dir, img_name)
            self._draw_research_style(img, complete_cs_poly, vb_polys, output_path)
            self.visualization_paths.append(output_path)

            # Record summary
            vb_pixel_area = sum([self._polygon_area(vb) for vb in vb_polys])
            cs_pixel_area = complete_cs_poly.area #now uses the combined polygon area
            
            areas = self._calculate_areas(vb_pixel_area, cs_pixel_area)
            ratio = self._calculate_ratio(areas["vb_area_microns"], areas["cs_area_microns"])

            records.append({
                "image_name": img_name,
                "vb_count": len(vb_polys),
                "vb_area_microns": round(areas["vb_area_microns"], 4),
                "cs_area_microns": round(areas["cs_area_microns"], 4),
                "vb_to_cs_ratio": round(ratio["vb_to_cs_ratio"], 4),
                "notes": ""
            })

        return pd.DataFrame(records)

    def _polygon_area(self, points):
        return Polygon(points).area

    def _polygon_center(self, points):
        poly = Polygon(points)
        return poly.centroid.x, poly.centroid.y

    def _calculate_areas(self, vb_pixel_area, cs_pixel_area):
        return {
            "vb_area_microns": vb_pixel_area * (self.PIXEL_TO_MICRON**2),
            "cs_area_microns": cs_pixel_area * (self.PIXEL_TO_MICRON**2)
        }

    def _calculate_ratio(self, vb_area, cs_area):
        ratio = vb_area / cs_area if cs_area > 0 else 0
        return {
            "vb_to_cs_ratio": ratio
        }

    def _create_complete_cross_section(self, main_poly, vb_polys):
        """
        Creates a complete cross section by combining the main polygon with vascular bundles.
        Uses robust geometry handling to avoid topology errors.
        """
        try:
            # Ensure main polygon is valid
            if not main_poly.is_valid:
                main_poly = make_valid(main_poly)
            
            # Collect all valid polygons
            all_polygons = [main_poly]
            
            for vb_points in vb_polys:
                try:
                    vb_poly = Polygon(vb_points)
                    if not vb_poly.is_valid:
                        vb_poly = make_valid(vb_poly)
                    
                    # Only add if it's a valid polygon after validation
                    if vb_poly.is_valid and not vb_poly.is_empty:
                        all_polygons.append(vb_poly)
                except Exception as e:
                    print(f"Warning: Skipping invalid vascular bundle polygon: {e}")
                    continue
            
            # Use unary_union for better handling of multiple polygons
            # This is more robust than iterative union operations
            if len(all_polygons) == 1:
                combined = all_polygons[0]
            else:
                combined = unary_union(all_polygons)
            
            # Ensure the result is valid
            if not combined.is_valid:
                combined = make_valid(combined)
            
            # Return the convex hull for a smooth outer boundary
            return combined.convex_hull
            
        except Exception as e:
            print(f"Warning: Error creating complete cross section, using main polygon only: {e}")
            # Fallback to just the main polygon if union fails
            return main_poly.convex_hull

    def _collect_polygons(self, result):
        preds = []
        for i, mask_polygon in enumerate(result.masks.xy):
            cls = int(result.boxes.cls[i])
            bbox = result.boxes[i]
            preds.append({
                "points": mask_polygon,
                "class_name": result.names[cls],
                "area": self._polygon_area(mask_polygon),
                "box": bbox.xyxy[0].cpu().numpy()
            })
        return preds

    def _find_main_cross_section(self, preds, img_shape):
        h, w = img_shape[:2]
        img_center = np.array([w / 2, h / 2])
        min_dist = float("inf")
        main_poly = None
        main_box = None

        for p in preds:
            if p["class_name"] == "Cross Section":
                cx, cy = self._polygon_center(p["points"])
                dist = np.linalg.norm(img_center - np.array([cx, cy]))
                if dist < min_dist:
                    min_dist = dist
                    main_poly = Polygon(p["points"])
                    main_box = p
        return main_poly, main_box

    def _filter_vascular_bundles(self, main_box, preds):
        main_cs_bbox = main_box['box']
        vb_polys = []
        for p in preds:
            if p["class_name"] == "Vascular Bundles":
                vb_center_x, vb_center_y = self._polygon_center(p["points"])
                if (vb_center_x >= main_cs_bbox[0] and vb_center_x <= main_cs_bbox[2] and
                    vb_center_y >= main_cs_bbox[1] and vb_center_y <= main_cs_bbox[3]):
                    vb_polys.append(p["points"])
        return vb_polys

    def _draw_research_style(self, img, main_poly, vb_polys, output_path):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if main_poly:
            x, y = main_poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=1.5)
        for vb in vb_polys:
            vb_poly = Polygon(vb)
            x, y = vb_poly.exterior.xy
            ax.fill(x, y, color="green", alpha=0.5)
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
