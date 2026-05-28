import gradio as gr
import os
import sys
import shutil
import tempfile
import gc
import pandas as pd
import zipfile
import json

_app_dir = os.path.dirname(os.path.abspath(__file__))
_third_party = os.path.abspath(os.path.join(_app_dir, "..", "third_party"))
_api_dir = os.path.join(_app_dir, "api")
for _p in (_third_party, _app_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(_api_dir)

from config.inference_constants import (
    CLASS_CASPARIAN,
    CLASS_CASPARIAN_FALLBACK,
    DEFAULT_CONF_MODEL_A,
    DEFAULT_CONF_MODEL_B,
    MODEL_A_CASPARIAN_EPIDERMIS,
    MODEL_B_VASCULAR_BUNDLES,
)
from converter import ImageConverter
from predictor import ModelPredictor
from dual_stem_pipeline import DualStemPipelineProcessor

# Per-run temp workspace: <temp>/input, converted, output
RUN_DIR_INPUT = "input"
RUN_DIR_CONVERTED = "converted"
RUN_DIR_OUTPUT = "output"
# output_images.zip top-level folders (stable for integrations)
ZIP_DIR_INPUT_IMAGES = "input_images"
ZIP_DIR_VISUALIZATIONS = "visualizations"
ZIP_DIR_LABELS = "labels_generated"
ZIP_DIR_GEOMETRY = "geometry_export"
ZIP_DIR_METRIC_DEBUG = "metric_debug_viz"
ZIP_DIR_DEBUG = "debug"
OUTPUT_CSV_NAME = "results.csv"
OUTPUT_BUNDLE_ZIP_NAME = "output_images.zip"
CONVERTED_BUNDLE_ZIP_NAME = "converted_images.zip"

RASTER_EXTENSIONS = (".png", ".jpg", ".jpeg")
GRADIO_FILE_TYPES = [".nd2", ".png", ".jpg", ".jpeg", ".zip"]

# Legacy single-model pipeline was removed from run_full_pipeline; see git history / post_processor.py.
# DEFAULT_MODEL_FILENAME = "best_v2.pt"
# DEFAULT_CONF_THRESHOLD = 0.80


def _extract_raw_prediction_records(predictions, source_label="main"):
    """Build JSON-serializable records from Ultralytics results (lightweight vs holding Results in RAM)."""
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
            elif class_name in (CLASS_CASPARIAN_FALLBACK, CLASS_CASPARIAN):
                entry["cross_sections"].append(item)
        records.append(entry)
    return records


def _write_raw_predictions_json(records, output_dir, source_label="main"):
    try:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_label)
        path = os.path.join(output_dir, f"raw_predictions_{safe}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _save_raw_predictions_debug(predictions, output_dir, source_label="main"):
    """Save raw model prediction summary (counts, VB/CS centers, bbox, conf, area) for debugging."""
    try:
        records = _extract_raw_prediction_records(predictions, source_label)
        _write_raw_predictions_json(records, output_dir, source_label)
    except Exception:
        pass  # do not break pipeline if debug save fails


def clear_all_analysis():
    """Clear all analysis inputs and outputs"""
    return None, "", None, None, pd.DataFrame()

def clear_all_conversion():
    """Clear all conversion inputs and outputs"""
    return None, "", None

def run_conversion(files, progress=gr.Progress()):
    """Conversion-only pipeline"""
    if not files:
        return None, "No files uploaded."
    
    progress(0, desc="Setting up...")
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, RUN_DIR_INPUT)
    converted_dir = os.path.join(temp_dir, RUN_DIR_CONVERTED)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(converted_dir, exist_ok=True)
    
    try:
        # Process uploaded files, extracting zip files
        progress(0.1, desc="Processing uploaded files...")
        for file in files:
            file_path = file.name
            if file_path.lower().endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for member in zip_ref.infolist():
                        sanitized_name = os.path.basename(member.filename)
                        # Skip directories and hidden/resource files (like those from macOS)
                        if member.is_dir() or sanitized_name.startswith('._') or sanitized_name.startswith('.'):
                            continue

                        if sanitized_name.lower().endswith(('.nd2',) + RASTER_EXTENSIONS):
                            with zip_ref.open(member) as source, open(os.path.join(input_dir, sanitized_name), 'wb') as target:
                                shutil.copyfileobj(source, target)
            elif file_path.lower().endswith(('.nd2',) + RASTER_EXTENSIONS):
                shutil.copy(file_path, os.path.join(input_dir, os.path.basename(file_path)))
            
        progress(0.3, desc="Converting images...")
        converter = ImageConverter()
        # Use converted_dir as the output directory for the converter
        png_files = converter.process_directory(input_dir, converted_dir)
        
        # The converted files are in converted_dir, so we zip that directory's contents
        # Also include any directly uploaded raster images (PNG/JPG/JPEG)
        for f in os.listdir(input_dir):
            if f.lower().endswith(RASTER_EXTENSIONS):
                shutil.copy(os.path.join(input_dir, f), converted_dir)

        progress(0.8, desc="Creating output archive...")

        # Zip the contents of the converted_dir
        output_zip_path = os.path.join(temp_dir, CONVERTED_BUNDLE_ZIP_NAME)
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for root, _, file_list in os.walk(converted_dir):
                for file in file_list:
                    zipf.write(os.path.join(root, file), file)
        
        num_images = len(os.listdir(converted_dir))
        if num_images == 0:
            return None, "No images were converted or uploaded."

        progress(1.0, desc="Complete!")
        status = f"Successfully converted/processed {num_images} images."
        return output_zip_path, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_full_pipeline(files, progress=gr.Progress()):
    """Main processing pipeline for analysis"""
    temp_dir = None
    if not files:
        return None, None, "No files uploaded", None, None
    
    try:
        # Setup temp directories
        progress(0, desc="Setting up directories...")
        temp_dir = tempfile.mkdtemp()
        input_dir = os.path.join(temp_dir, RUN_DIR_INPUT)
        converted_dir = os.path.join(temp_dir, RUN_DIR_CONVERTED)
        output_dir = os.path.join(temp_dir, RUN_DIR_OUTPUT)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(converted_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Process uploaded files, extracting zip files
        progress(0.1, desc="Processing uploaded files...")
        for file in files:
            file_path = file.name
            if file_path.lower().endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for member in zip_ref.infolist():
                        sanitized_name = os.path.basename(member.filename)
                        # Skip directories and hidden/resource files (like those from macOS)
                        if member.is_dir() or sanitized_name.startswith('._') or sanitized_name.startswith('.'):
                            continue

                        if sanitized_name.lower().endswith(('.nd2',) + RASTER_EXTENSIONS):
                            with zip_ref.open(member) as source, open(os.path.join(input_dir, sanitized_name), 'wb') as target:
                                shutil.copyfileobj(source, target)
            elif file_path.lower().endswith(('.nd2',) + RASTER_EXTENSIONS):
                shutil.copy(file_path, os.path.join(input_dir, os.path.basename(file_path)))
        
        # Convert ND2 files
        progress(0.2, desc="Converting ND2 files to PNG...")
        converter = ImageConverter()
        converted_pngs = converter.process_directory(input_dir, converted_dir)
        
        # Also include any direct PNG/JPG/JPEG uploads
        direct_raster_images = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(RASTER_EXTENSIONS)
        ]
        
        # All raster images for processing are in converted_dir or input_dir
        all_images_for_processing = converted_pngs + direct_raster_images

        if not all_images_for_processing:
            return None, None, "No images to process", None, temp_dir
        
        # Run predictions
        progress(0.4, desc=f"Running detection on {len(all_images_for_processing)} images...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        debug_output_dir = os.path.join(project_root, "debug_output")
        os.makedirs(debug_output_dir, exist_ok=True)

        candidate_model_dirs = [
            os.path.join(project_root, "models"),
            os.path.join(project_root, "sample_trained_models"),
            os.path.join(os.path.dirname(script_dir), "models"),
            os.path.join(os.path.dirname(script_dir), "sample_trained_models"),
        ]

        models_dir = None

        for candidate in candidate_model_dirs:
            path_a = os.path.join(candidate, MODEL_A_CASPARIAN_EPIDERMIS)
            path_b = os.path.join(candidate, MODEL_B_VASCULAR_BUNDLES)
            if os.path.isfile(path_a) and os.path.isfile(path_b):
                models_dir = candidate
                break

        if models_dir is None:
            searched = "\n".join(candidate_model_dirs)
            msg = (
                "Missing model weights. Looked in these folders:\n"
                + searched
                + "\n\nExpected files:\n"
                + MODEL_A_CASPARIAN_EPIDERMIS
                + "\n"
                + MODEL_B_VASCULAR_BUNDLES
            )
            return None, None, msg, None, temp_dir

        print(f"Using models from: {models_dir}")
        print(f"Model A: {path_a}")
        print(f"Model B: {path_b}")

        n_img = len(all_images_for_processing)
        predictor_a = ModelPredictor(path_a)
        predictor_b = ModelPredictor(path_b)
        processor = DualStemPipelineProcessor(output_dir, debug_log_dir=debug_output_dir)

        all_dfs = []
        all_viz: list[str] = []
        all_records_a: list = []
        all_records_b: list = []

        for idx, path in enumerate(all_images_for_processing):
            progress(
                0.35 + 0.35 * (idx / max(n_img, 1)),
                desc=f"Image {idx + 1}/{n_img}: models + postprocess...",
            )
            results_a = predictor_a.batch_predict([path], output_dir, DEFAULT_CONF_MODEL_A, save=False)
            results_b = predictor_b.batch_predict([path], output_dir, DEFAULT_CONF_MODEL_B, save=False)
            all_records_a.extend(_extract_raw_prediction_records(results_a, "model_a"))
            all_records_b.extend(_extract_raw_prediction_records(results_b, "model_b"))
            df_i = processor.process_batch(
                [path], results_a, results_b, append_debug_log=(idx > 0)
            )
            all_dfs.append(df_i)
            all_viz.extend(processor.visualization_paths)
            del results_a, results_b
            gc.collect()

        _write_raw_predictions_json(all_records_a, debug_output_dir, source_label="model_a")
        _write_raw_predictions_json(all_records_b, debug_output_dir, source_label="model_b")

        progress(0.7, desc="Assembling results...")
        df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        viz_files = all_viz
        
        # Save CSV
        csv_path = None
        if not df.empty:
            csv_path = os.path.join(output_dir, OUTPUT_CSV_NAME)
            df.to_csv(csv_path, index=False)
        
        # Create a zip file with all the output images and debug files
        progress(0.9, desc="Creating output archive...")
        output_zip_path = os.path.join(output_dir, OUTPUT_BUNDLE_ZIP_NAME)
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            # Add converted images
            for input_image in all_images_for_processing:
                zipf.write(
                    input_image,
                    os.path.join(ZIP_DIR_INPUT_IMAGES, os.path.basename(input_image)),
                )
            # Add post-processor visualizations
            for viz_file in viz_files:
                zipf.write(
                    viz_file,
                    os.path.join(ZIP_DIR_VISUALIZATIONS, os.path.basename(viz_file)),
                )
            labels_dir = os.path.join(output_dir, ZIP_DIR_LABELS)
            if os.path.isdir(labels_dir):
                for fn in os.listdir(labels_dir):
                    fp = os.path.join(labels_dir, fn)
                    if os.path.isfile(fp):
                        zipf.write(fp, os.path.join(ZIP_DIR_LABELS, fn))
            geom_dir = os.path.join(output_dir, ZIP_DIR_GEOMETRY)
            if os.path.isdir(geom_dir):
                for fn in os.listdir(geom_dir):
                    fp = os.path.join(geom_dir, fn)
                    if os.path.isfile(fp):
                        zipf.write(fp, os.path.join(ZIP_DIR_GEOMETRY, fn))
            # Per-metric debug PNGs (export_metric_debug_visualizations in stem_metrics)
            metric_debug_dir = os.path.join(output_dir, ZIP_DIR_METRIC_DEBUG)
            if os.path.isdir(metric_debug_dir):
                for root, _dirs, files in os.walk(metric_debug_dir):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        if os.path.isfile(fp):
                            rel = os.path.relpath(fp, metric_debug_dir)
                            arcname = os.path.join(ZIP_DIR_METRIC_DEBUG, rel).replace("\\", "/")
                            zipf.write(fp, arcname)
            # Add debug files from project folder so they are in the zip too
            log_path = os.path.join(debug_output_dir, "post_processor_debug.log")
            if os.path.isfile(log_path):
                zipf.write(log_path, f"{ZIP_DIR_DEBUG}/post_processor_debug.log")
            for pat in ("raw_predictions_model_a.json", "raw_predictions_model_b.json"):
                raw_json = os.path.join(debug_output_dir, pat)
                if os.path.isfile(raw_json):
                    zipf.write(raw_json, os.path.join(ZIP_DIR_DEBUG, pat))

        status = f"Processed {len(all_images_for_processing)} images.\n"
        if not df.empty:
            status += f"Found {len(df)} detections.\n"
        
        progress(1.0, desc="Complete!")
        return csv_path, output_zip_path, status, df.head(50) if not df.empty else pd.DataFrame(), temp_dir
        
    except Exception as e:
        import traceback
        return None, None, f"Error: {str(e)}\n{traceback.format_exc()}", None, temp_dir


# Build UI with Tabs
with gr.Blocks(title="Alfalfa Stem Tool") as app:
    gr.Markdown("# Alfalfa Stem Object Detection Tool")

    with gr.Tabs():
        with gr.TabItem("Full Detection Pipeline"):
            with gr.Row():
                with gr.Column(scale=2):
                    analysis_files = gr.File(
                        label="Upload ND2, PNG, JPG, or ZIP · RR or PG in filename",
                        file_count="multiple",
                        file_types=GRADIO_FILE_TYPES,
                    )
                    analysis_btn = gr.Button("Run Analysis", variant="primary")
                
                with gr.Column(scale=3):
                    analysis_status = gr.Textbox(label="Status", lines=5)
            
            gr.Markdown("### Download Results")
            with gr.Row():
                analysis_csv = gr.File(label="Results CSV")
                analysis_zip = gr.File(label="Output Images (ZIP)")
            
            gr.Markdown("### Results Preview")
            analysis_preview = gr.Dataframe(label="Detection Results")

            # Clear everything button at the bottom
            analysis_clear_all_btn = gr.Button("Clear All & Start New Analysis", variant="stop", size="lg")

        with gr.TabItem("Simple Image Converter"):
            with gr.Row():
                with gr.Column(scale=2):
                    convert_files = gr.File(
                        label="Upload ND2, PNG, JPG, or ZIP",
                        file_count="multiple",
                        file_types=GRADIO_FILE_TYPES,
                    )
                    convert_btn = gr.Button("Convert", variant="primary")
                with gr.Column(scale=3):
                    convert_status = gr.Textbox(label="Status", lines=3)
            
            gr.Markdown("### Download Converted Images")
            convert_zip = gr.File(label="Converted Images (ZIP)")

            # Clear everything button at the bottom
            convert_clear_all_btn = gr.Button("Clear All & Start New Conversion", variant="stop", size="lg")

    # Hook up Analysis tab functions
    analysis_btn.click(
        fn=run_full_pipeline,
        inputs=[analysis_files],
        outputs=[analysis_csv, analysis_zip, analysis_status, analysis_preview],
        api_name="run_analysis"
    )
    
    # Clear everything in analysis tab
    analysis_clear_all_btn.click(
        fn=clear_all_analysis,
        inputs=[],
        outputs=[analysis_files, analysis_status, analysis_csv, analysis_zip, analysis_preview]
    )

    # Hook up Conversion tab functions
    convert_btn.click(
        fn=run_conversion,
        inputs=[convert_files],
        outputs=[convert_zip, convert_status],
        api_name="run_conversion"
    )

    # Clear everything in conversion tab
    convert_clear_all_btn.click(
        fn=clear_all_conversion,
        inputs=[],
        outputs=[convert_files, convert_status, convert_zip]
    )


if __name__ == "__main__":
    # Allow `python main.py` to launch the UI directly (mirrors app.py behavior).
    # Port resolution: GRADIO_SERVER_PORT, then PORT, then 7860.
    _port = next(
        (int(os.environ[k]) for k in ("GRADIO_SERVER_PORT", "PORT") if os.environ.get(k)),
        7860,
    )
    app.queue().launch(server_port=_port, inbrowser=True)
