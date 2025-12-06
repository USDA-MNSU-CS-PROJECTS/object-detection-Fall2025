import os
from ultralytics import YOLO
import torch
import requests
from datetime import datetime

DATA_CONFIG = "data.yaml"
MODEL_NAME = "yolov8n-seg.pt"
OUTPUT_DIR = "runs/train/exp_test"

DISCORD_WEBHOOK_URL = "https://canary.discord.com/api/webhooks/1432756517110550538/rXtt-rFfpdn-kmvZmOVhEZMBwfscCBXzZxErnbsnb_hRGFy_N7qYMydCALvUD1DrcIXT"

def send_discord_notification(message, title="Training Update"):
    """Send notification to Discord"""
    print(f"DEBUG: Sending Discord notification: {title}")
    try:
        data = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": 5814783,  
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {
                    "text": "YOLOv8 Training Bot"
                }
            }]
        }
        
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        
        if response.status_code == 204:
            print("Discord notification sent")
        else:
            print(f"Discord notification failed: {response.status_code}")
            
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def main():
    print(f"Using dataset: {DATA_CONFIG}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    send_discord_notification(
        f"Training started!\n"
        f"Dataset: {DATA_CONFIG}\n"
        f"Model: {MODEL_NAME}\n"
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        title="Training Started"
    )
    model = YOLO(MODEL_NAME)

    try:
        print("DEBUG: Starting training")
        results = model.train(
            data=DATA_CONFIG,
            epochs=400,       
            imgsz=640,
            batch=16,
            device=2,
            project="runs/train",
            name="exp_final_confidence_push",
            exist_ok=True,
            workers=8,
            patience=200,      
            
            #Confidence boosting:
            cls=4.0,           
            box=7.5,
            dfl=1.5,
            label_smoothing=0.0,
            
            #Augmentation for confidence:
            hsv_h=0.003,       
            hsv_s=0.2,         
            hsv_v=0.15,        
            degrees=2,         
            translate=0.02,    
            scale=0.15,        
            fliplr=0.5,        
            mosaic=0.3,        
            mixup=0.0,         
            copy_paste=0.0,    
            dropout=0.0,
            
            #Fine-tuning learning rate:
            lr0=0.005,  
            lrf=0.005,
            warmup_epochs=15,
            close_mosaic=20,  
            cos_lr=True,  
        )
        
        print("DEBUG: Training completed, sending notification")
        send_discord_notification(
            f"Training completed successfully!\n"
            f"Results saved to: {OUTPUT_DIR}\n"
            f"Check your model weights in the runs folder!",
            title="Training Complete!"
        )
        
        print(f"Training complete. Check output in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"DEBUG: Exception caught: {e}")
        # Send error notification
        send_discord_notification(
            f"Training failed with error:\n```{str(e)}```",
            title="Training Failed"
        )
        print(f"Training failed: {e}")
        raise e

print("DEBUG: About to call main()")
if __name__ == "__main__":
    main()
print("DEBUG: Script finished")
