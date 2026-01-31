# Object Detection using YOLOv5, SSD, Faster R-CNN & RetinaNet

Application Python de détection d’objets qui permet de tester/Comparer plusieurs familles de modèles :
- **YOLOv5** (one-stage, rapide)
- **SSD** (one-stage)
- **Faster R-CNN** (two-stage, souvent plus précis mais plus lourd)
- **RetinaNet** (one-stage avec focal loss)

Le projet contient une application principale `app.py`, des helpers (`helpers.py`) et un chargeur de modèles (`model_loader.py`). :contentReference[oaicite:1]{index=1}

---

## Structure du projet

- `app.py` : point d’entrée de l’application :contentReference[oaicite:2]{index=2}  
- `model_loader.py` : chargement/initialisation des modèles :contentReference[oaicite:3]{index=3}  
- `helpers.py` : fonctions utilitaires (pré/post-traitement, affichage, etc.) :contentReference[oaicite:4]{index=4}  
- `requirements.txt` : dépendances Python :contentReference[oaicite:5]{index=5}  
- `assets/` : ressources (images, exemples, etc.) :contentReference[oaicite:6]{index=6}  
- `yolov5s.pt` : poids YOLOv5 (fourni dans le repo) :contentReference[oaicite:7]{index=7}  

---

## Installation

### 1) Cloner le repo
```bash
git clone https://github.com/anlehtouf/Object-Detection-using-YOLOv5-SSD-Faster-R-CNN-RetinaNet.git
cd Object-Detection-using-YOLOv5-SSD-Faster-R-CNN-RetinaNet
