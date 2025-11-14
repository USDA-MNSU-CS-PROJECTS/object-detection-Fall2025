from ultralytics import YOLO

class ModelPredictor:
    def __init__(self, model_path: str):
        """
        Initializes the ModelPredictor.

        Args:
            model_path (str): The path to the YOLO model.
        """
        self.model = YOLO(model_path)
    
    def predict(self, image_path: str, conf_threshold: float, save: bool = False, output_dir: str = None, imgsz: int = 640) -> list:
        """
        Runs prediction on a single image.

        Args:
            image_path (str): Path to the image.
            conf_threshold (float): Confidence threshold for predictions.
            save (bool): Whether to save images with overlays.
            output_dir (str): Directory to save images if save is True.
            imgsz (int): Image size for prediction.

        Returns:
            list: A list containing a single ultralytics result object.
        """
        if save and not output_dir:
            raise ValueError("output_dir must be provided if save is True.")
            
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=save,
            project=output_dir if save else None,
            name="predicted_images",
            exist_ok=True,
            verbose=False,
            imgsz=imgsz
        )
        return results

    def batch_predict(self, image_paths: list, output_dir: str, conf_threshold: float, save: bool = False, imgsz: int = 640) -> list:
        """
        Runs prediction on a batch of images.

        Args:
            image_paths (list): List of paths to images.
            output_dir (str): Directory to save images if save is True.
            conf_threshold (float): Confidence threshold for predictions.
            save (bool): Whether to save images with overlays.
            imgsz (int): Image size for prediction.

        Returns:
            list: A list of ultralytics result objects.
        """
        results = self.model.predict(
            source=image_paths,
            conf=conf_threshold,
            save=save,
            project=output_dir if save else None,
            name="predicted_images",
            exist_ok=True,
            verbose=False,
            imgsz=imgsz
        )
        return results
