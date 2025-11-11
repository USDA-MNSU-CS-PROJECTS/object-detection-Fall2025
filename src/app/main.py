import gradio as gr
import os
import sys
import shutil
import tempfile
import pandas as pd

# Add api folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

# Import what's ready
from converter import ImageConverter

# Mock up what's not ready yet
try:
    from predictor import ModelPredictor
    PREDICTOR_READY = True
except:
    PREDICTOR_READY = False
    print("ModelPredictor not ready - using mock")

try:
    from post_processor import PostProcessor
    POSTPROCESSOR_READY = True
except:
    POSTPROCESSOR_READY = False
    print("PostProcessor not ready - using mock")


def mock_predict(image_paths, conf_threshold):
    """Temporary mock - replace when predictor.py is done"""
    import random
    predictions = []
    for img_path in image_paths:
        predictions.append({
            'image_name': os.path.basename(img_path),
            'total_detections': random.randint(0, 5),
            'detections': [
                {
                    'class_name': 'test_object',
                    'confidence': 0.85,
                    'bbox_x1': 100, 'bbox_y1': 100,
                    'bbox_x2': 200, 'bbox_y2': 200
                }
            ]
        })
    return predictions


def mock_postprocess(predictions):
    """Temporary mock - replace when post_processor.py is done"""
    rows = []
    for pred in predictions:
        for det in pred.get('detections', []):
            rows.append({
                'image_name': pred['image_name'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox_x1': det['bbox_x1'],
                'bbox_y1': det['bbox_y1']
            })
    return pd.DataFrame(rows)


def process_images(files, conf_threshold):
    """Main processing pipeline"""
    
    if not files:
        return None, None, "No files uploaded"
    
    # Setup temp directories
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, "input")
    converted_dir = os.path.join(temp_dir, "converted")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save uploaded files
        for file in files:
            shutil.copy(file.name, os.path.join(input_dir, os.path.basename(file.name)))
        
        # Convert ND2 files (REAL converter - it's done!)
        converter = ImageConverter()
        png_files = converter.process_directory(input_dir, converted_dir)
        
        # Also include any direct PNG uploads
        png_files.extend([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                         if f.lower().endswith('.png')])
        
        if not png_files:
            return None, None, "No images to process", None
        
        # Run predictions
        if PREDICTOR_READY:
            predictor = ModelPredictor("models/best.pt")
            predictions = predictor.batch_predict(png_files, output_dir, conf_threshold)
        else:
            predictions = mock_predict(png_files, conf_threshold)
        
        # Post-process
        if POSTPROCESSOR_READY:
            processor = PostProcessor()
            df = processor.process_predictions(predictions)
        else:
            df = mock_postprocess(predictions)
        
        # Save CSV
        csv_path = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        
        # Create a zip file with all the PNGs
        import zipfile
        png_zip_path = os.path.join(output_dir, "converted_images.zip")
        with zipfile.ZipFile(png_zip_path, 'w') as zipf:
            for png_file in png_files:
                zipf.write(png_file, os.path.basename(png_file))
        
        status = f"Processed {len(png_files)} images\n"
        status += f"Found {len(df)} detections\n"
        if not PREDICTOR_READY:
            status += "Using mock predictions (predictor not ready)\n"
        if not POSTPROCESSOR_READY:
            status += "Using mock post-processing\n"
        
        return csv_path, png_zip_path, status, df.head(50)
        
    except Exception as e:
        return None, None, f"Error: {str(e)}", None


# Build UI
with gr.Blocks(title="Alfalfa Stem Object Detection") as app:
    gr.Markdown("# Object Detection Tool")
    
    with gr.Row():
        with gr.Column():
            files = gr.File(label="Upload ND2 or PNG files", file_count="multiple")
            conf = gr.Slider(0.1, 0.9, value=0.25, label="Confidence Threshold")
            btn = gr.Button("Process", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status", lines=5)
    
    gr.Markdown("### Download Results")
    
    with gr.Row():
        with gr.Column():
            csv_output = gr.File(label="Results CSV")
        with gr.Column():
            png_output = gr.File(label="Converted PNGs (ZIP)")
    
    gr.Markdown("### Preview")
    preview = gr.Dataframe(label="Detection Results")
    
    btn.click(
        fn=process_images,
        inputs=[files, conf],
        outputs=[csv_output, png_output, status, preview],
        api_name="process"
    )


if __name__ == "__main__":
    app.launch(inbrowser=True)  # Automatically opens browser