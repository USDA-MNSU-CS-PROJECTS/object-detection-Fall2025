import gradio as gr
from api import ImageConverter, ModelPredictor, PostProcessor

def create_app():
    def process_folder(input_folder):
        # 1. Convert images
        # 2. Run predictions
        # 3. Post-process
        # 4. Return results
        pass

    with gr.Blocks() as app:
        gr.Markdown("# Cell Image Analysis")
        with gr.Row():
            input_folder = gr.File(label="Input Folder", file_count="directory")
            output = gr.File(label="Output CSV")
        
        submit_btn = gr.Button("Process Images")
        submit_btn.click(fn=process_folder, 
                        inputs=[input_folder],
                        outputs=[output])
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()