from ultralytics import YOLO
class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def predict(self, image_path: str, output_dir: str) -> dict:
        # Wrapper for model prediction
        pass

    def batch_predict(self, image_paths: list, output_dir: str) -> list:
        # Handle multiple images
        pass