# model_loader.py
import torch
from PIL import Image
import numpy as np

def load_model():
    # Charge YOLOv5s directement depuis PyTorch Hub
    # Pas besoin d'installer yolov5 en tant que package !
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model.eval()
    return model

def run_inference(model, image: Image.Image, conf_threshold=0.25):
    img_rgb = np.array(image)
    if img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    # Désactiver les logs verbeux
    model.conf = conf_threshold
    results = model(img_rgb)

    annotated_img = results.render()[0]  # BGR numpy array
    detections = results.xyxy[0].cpu().numpy()
    detections = detections[detections[:, 4] >= conf_threshold]

    return annotated_img, detections