import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# Flask app initialization
app = Flask(__name__)
CORS(app, origins="*") # Enable CORS

# Classes to detect (COCO dataset classes)
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
               'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
               'teddy bear', 'hair drier', 'toothbrush']

# Target classes to filter
TARGET_CLASSES = ['banana', 'apple', 'orange', 'broccoli', 'carrot']

# Custom configuration class
class InferenceConfig(Config):
    NAME = "object_detection"
    NUM_CLASSES = len(CLASS_NAMES)  # Background + all classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Load Mask R-CNN model
config = InferenceConfig()
model = MaskRCNN(mode="inference", model_dir=".", config=config)

# Load the weights for COCO
model_path = "mask_rcnn_coco.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Mask R-CNN weights file not found.")

model.load_weights(model_path, by_name=True)

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400

    # Load and preprocess the image
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image = np.array(image)

    # Perform detection
    results = model.detect([image], verbose=0)
    r = results[0]

    detected_objects = []
    for i in range(len(r['class_ids'])):
        class_id = r['class_ids'][i]
        class_name = CLASS_NAMES[class_id]

        if class_name in TARGET_CLASSES and r['scores'][i] > 0.5:  # Filter by class and confidence score
            detected_objects.append({
                "class_id": int(class_id),
                "name": class_name,
                "score": float(r['scores'][i]),
                "bbox": [
                    int(r['rois'][i][1]),  # x1
                    int(r['rois'][i][0]),  # y1
                    int(r['rois'][i][3]),  # x2
                    int(r['rois'][i][2])   # y2
                ]
            })

    # Return detection results
    return jsonify({"detected_objects": detected_objects})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
