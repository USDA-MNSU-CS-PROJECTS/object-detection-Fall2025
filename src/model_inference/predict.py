from ultralytics import YOLO

# Load the trained segmentation model
model = YOLO('sample_trained_models/best_multi_class_client_hpc.pt')

# Run prediction on your local images
model.predict(
    source='test_images',   # folder of images or single image path
    save=True,
    imgsz=640,
    conf=0.50
)
