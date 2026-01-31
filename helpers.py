# utils.py
import cv2
from PIL import Image
import numpy as np
def cv2_to_pil(cv2_img):
    """Convertit une image OpenCV (BGR) en PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """Convertit une image PIL (RGB) en OpenCV (BGR)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)