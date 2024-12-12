# RUN BELOW FIRST THEN RUN BELOW

# First run the code to get the validation metrics of best.pt

from ultralytics import YOLO

#model_path = "/content/runs/detect/train/weights/best.pt"  # your trained model
#data_yaml_path = "/content/bccd_rbc_wbc_platelets-1/data.yaml"  # dataset configuration file

#model = YOLO(model_path)
#metrics = model.val(data=data_yaml_path)  # ensure data.yaml points to the correct valid set

# Extract overall metrics
#overall_precision = metrics.box.mp       # mean precision over all classes
#overall_recall = metrics.box.mr          # mean recall over all classes
#overall_map50 = metrics.box.map50        # mean AP at IoU=0.5 over all classes
#overall_map = metrics.box.map            # mean AP at IoU=0.5:0.95 over all classes
#overall_map75 = metrics.box.map75        # mean AP at IoU=0.75 over all classes

# Extract per-class metrics
#class_names = model.names  # or load from data.yaml if needed, same as model.names
#class_metrics = []
#for i, cname in enumerate(class_names):
#    p, r, ap50, ap = metrics.box.class_result(i)
#    class_metrics.append((cname, p, r, ap50, ap))

#print("Overall Metrics:")
#print(f"Precision: {overall_precision}")
#print(f"Recall: {overall_recall}")
#print(f"mAP50: {overall_map50}")
#print(f"mAP50-95: {overall_map}")
#print(f"mAP75: {overall_map75}")
#print("\nPer-Class Metrics:")
#for (cname, p, r, ap50, ap) in class_metrics:
#    print(f"{cname}: Precision={p}, Recall={r}, mAP50={ap50}, mAP50-95={ap}")



############ Take the values from abover and put them below manually


############## Use below for production with manual metrics input


import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import yaml
import gradio as gr
import pandas as pd

model_path = "best.pt"   # Keep best.pt on same directory
data_yaml_path = "data.yaml"  # Keep data.yaml on same directory

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}.")

# Load the YOLO model
model = YOLO(model_path)

# Load class names
with open(data_yaml_path, 'r') as stream:
    data_dict = yaml.safe_load(stream)
class_names = data_dict['names']  # e.g., ['Platelets', 'RBC', 'WBC'] if those are your classes

##################################
# Hardcoded metrics from your provided values:
overall_precision = 0.8998657967724281
overall_recall = 0.9152413015416975
overall_map50 = 0.9482967626275897
overall_map = 0.6529025986330599
overall_map75 = 0.7199225312247104

# Per-Class Metrics (index as per data.yaml order)
# Here we assume the class order matches the indices: 
# class_names[0], class_names[1], class_names[2], etc.
class0_precision = 0.8820047185253768
class0_recall = 0.958904109589041
class0_map50 = 0.961433378998409
class0_map = 0.5023530432704303

class1_precision = 0.8196728808767741
class1_recall = 0.7868197950360514
class1_map50 = 0.8884569088843599
class1_map = 0.6302822447945686

class2_precision = 0.9979197909151334
class2_recall = 1.0
class2_map50 = 0.995
class2_map = 0.8260725078341811

# Construct the metrics DataFrame
metrics_data = [
    ["Overall", overall_precision, overall_recall, overall_map50, overall_map],
    [class_names[0], class0_precision, class0_recall, class0_map50, class0_map],
    [class_names[1], class1_precision, class1_recall, class1_map50, class1_map],
    [class_names[2], class2_precision, class2_recall, class2_map50, class2_map]
]
metrics_df = pd.DataFrame(metrics_data, columns=["Class", "Precision", "Recall", "mAP50", "mAP50-95"])
##################################

def run_inference(img: np.ndarray, model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb, conf=0.25, iou=0.6)
    detections = []
    res = results[0]
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_idx = int(boxes.cls[i])
            class_name = class_names[cls_idx]
            detections.append([class_name, conf, *xyxy])
    return detections

def draw_boxes(image: np.ndarray, detections):
    # Define a color palette for classes (BGR)
    palette = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
    ]
    num_colors = len(palette)

    for det in detections:
        class_name, conf, x1, y1, x2, y2 = det
        cls_idx = class_names.index(class_name)
        color = palette[cls_idx % num_colors]

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Text settings
        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        # Draw filled rectangle behind text
        cv2.rectangle(image, (int(x1), int(y1)-th-8), (int(x1)+tw, int(y1)), color, -1)
        # Put text in white for visibility
        cv2.putText(image, label, (int(x1), int(y1)-5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image

def process_image(image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detections = run_inference(img_bgr, model)

    annotated_img = draw_boxes(img_bgr.copy(), detections)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    det_df = pd.DataFrame(detections, columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])

    # Return annotated image, detection results, and hardcoded metrics table
    return Image.fromarray(annotated_img_rgb), det_df, metrics_df

with gr.Blocks() as demo:
    gr.Markdown("# YOLOn11 Cell Detection Web App")
    gr.Markdown("Upload an image and the model will return bounding boxes, classes, and confidence scores.")
    gr.Markdown("Metrics shown below are pre-computed and hardcoded into the code.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Run Inference")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Annotated Image")
            det_results = gr.DataFrame(label="Detection Results")
            metrics_table = gr.DataFrame(value=metrics_df, label="Validation Metrics")

    submit_btn.click(fn=process_image, inputs=input_image, outputs=[output_image, det_results, metrics_table])

demo.launch()

