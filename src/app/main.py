import gradio as gr
import os
import sys
import shutil
import tempfile
import pandas as pd
import zipfile

# Add api folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

# Import APIs
from converter import ImageConverter
from predictor import ModelPredictor
from post_processor import PostProcessor

# Configuration for the default model
DEFAULT_MODEL_FILENAME = "C:/Users/Abi/Documents/GitHub/Dave-bot/sample_trained_models/best.pt"
DEFAULT_CONF_THRESHOLD = 0.80


def clear_all_analysis():
    """Clear all analysis inputs and outputs"""
    return None, "", None, None, pd.DataFrame()

def clear_all_conversion():
    """Clear all conversion inputs and outputs"""
    return None, "", None

def run_conversion(files):
    """Conversion-only pipeline"""
    if not files:
        return None, "No files uploaded."
    
    progress(0, desc="Setting up...")
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, "input")
    converted_dir = os.path.join(temp_dir, "converted")
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

                        if sanitized_name.lower().endswith(('.nd2', '.png')):
                            with zip_ref.open(member) as source, open(os.path.join(input_dir, sanitized_name), 'wb') as target:
                                shutil.copyfileobj(source, target)
            elif file_path.lower().endswith(('.nd2', '.png')):
                shutil.copy(file_path, os.path.join(input_dir, os.path.basename(file_path)))
            
        progress(0.3, desc="Converting images...")
        converter = ImageConverter()
        # Use converted_dir as the output directory for the converter
        png_files = converter.process_directory(input_dir, converted_dir)
        
        # The converted files are in converted_dir, so we zip that directory's contents
        # Also include any PNGs that were uploaded directly
        for f in os.listdir(input_dir):
            if f.lower().endswith('.png'):
                shutil.copy(os.path.join(input_dir, f), converted_dir)

        progress(0.8, desc="Creating output archive...")

        # Zip the contents of the converted_dir
        output_zip_path = os.path.join(temp_dir, "converted_images.zip")
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
        input_dir = os.path.join(temp_dir, "input")
        converted_dir = os.path.join(temp_dir, "converted")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(converted_dir, exist_ok=True)

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

                        if sanitized_name.lower().endswith(('.nd2', '.png')):
                            with zip_ref.open(member) as source, open(os.path.join(input_dir, sanitized_name), 'wb') as target:
                                shutil.copyfileobj(source, target)
            elif file_path.lower().endswith(('.nd2', '.png')):
                shutil.copy(file_path, os.path.join(input_dir, os.path.basename(file_path)))
        
        # Convert ND2 files
        progress(0.2, desc="Converting ND2 files to PNG...")
        converter = ImageConverter()
        converted_pngs = converter.process_directory(input_dir, converted_dir)
        
        # Also include any direct PNG uploads
        direct_pngs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        
        # All PNGs for processing are in converted_dir or input_dir
        all_pngs_for_processing = converted_pngs + direct_pngs

        if not all_pngs_for_processing:
            return None, None, "No images to process", None, temp_dir
        
        # Run predictions
        progress(0.4, desc=f"Running detection on {len(all_pngs_for_processing)} images...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        model_path = os.path.join(project_root, "sample_trained_models", DEFAULT_MODEL_FILENAME)
        predictor = ModelPredictor(model_path)
        predictions = predictor.batch_predict(all_pngs_for_processing, output_dir, DEFAULT_CONF_THRESHOLD, save=False)
        
        # Post-process
        progress(0.7, desc="Processing results and creating visualizations...")
        processor = PostProcessor(output_dir)
        df = processor.process_predictions(predictions)
        viz_files = processor.visualization_paths
        
        # Save CSV
        csv_path = None
        if not df.empty:
            csv_path = os.path.join(output_dir, "results.csv")
            df.to_csv(csv_path, index=False)
        
        # Create a zip file with all the output images
        progress(0.9, desc="Creating output archive...")
        output_zip_path = os.path.join(output_dir, "output_images.zip")
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            # Add converted images
            for png_file in all_pngs_for_processing:
                zipf.write(png_file, os.path.join('input_images', os.path.basename(png_file)))
            # Add post-processor visualizations
            for viz_file in viz_files:
                zipf.write(viz_file, os.path.join('visualizations', os.path.basename(viz_file)))

        status = f"Processed {len(all_pngs_for_processing)} images.\n"
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
                    analysis_files = gr.File(label="Upload ND2, PNG, or ZIP files", file_count="multiple", file_types=[".nd2", ".png", ".zip"])
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
                    convert_files = gr.File(label="Upload ND2, PNG, or ZIP files", file_count="multiple", file_types=[".nd2", ".png", ".zip"])
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
    app.queue() # For progress tracking
    app.launch(inbrowser=True)
