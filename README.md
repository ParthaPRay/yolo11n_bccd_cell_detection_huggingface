# yolo11n_bccd_cell_detection
This repo contains code and instruction to fine tune yolo11n and with bccd dataset for RBC, WBC and Platelets cell object detection and hosting the fine tuned model on huggingface space

# Yolon11 Fine Tuning for BCCD Dataset object detection

# Dataset

https://github.com/Shenggan/BCCD_Dataset

BCCD Dataset is a small-scale dataset for blood cells detection.

BCCD has three kind of labels :

- RBC
- WBC
- Platelets

Below image is retrieved from the above dataset ![image](https://github.com/user-attachments/assets/8963ce8b-8fb4-4611-8ae7-3055080c6cb6)

# First Use Roboflow on the above dataset for proceprocessing and augmentation

On Roboflow make a project

Then

**Image Preparation**

Download the above dataset into local machine.


Then unzip.

Then cd BCCD_Dataset-master/BCCD/JPEGImages

- Then select all images of the folder and upload to the Roboflow project.

Then cd BCCD_Dataset-master/BCCD/Annotations

- Then select all images of the folder and upload to the Roboflow project.

It will **automate** the image annotation by Roboflow.

Then 

**Preprocessing and Augmentation**

Do Prepocessing 

- Auto-Orient: Applied
- Resize: Stretch to 640x480

Then 

Do Augmentation

- Outputs per training example: 3
- Flip: Horizontal, Vertical
- Crop: 0% Minimum Zoom, 20% Maximum Zoom
- Rotation: Between -15° and +15°
- Hue: Between -15° and +15°
- Saturation: Between -25% and +25%
- Brightness: Between -15% and +15%
- Blur: Up to 2.5px
- Noise: Up to 0.1% of pixels

Doing so will increase the total number image to 874

Dataset Split

- Train Set: 88%: 765 Images
- Valid Set: 8%: 73 Images
- Test Set: 4%: 36 Images

![image](https://github.com/user-attachments/assets/c921f83c-965c-4e1e-bb78-b07a28323d8a)

---

# Roboflow project download for Yolo11

Then select Roboflow project for Yolo11 as below

![image](https://github.com/user-attachments/assets/45e8136d-4d57-4d50-80c0-b7bf2d167d35)

It will allow to download the above projct by below code

![image](https://github.com/user-attachments/assets/3a4b9705-cf66-4b97-8f70-e5890a124b8a)

The dataset directory might look like as below

![image](https://github.com/user-attachments/assets/f7d57ec5-1367-448c-adc1-3639de0309c7)

Modify data.yaml as per below for train, val and test. nc is for number of classes i.e. 3. list of claases start with 0, 1, and 2 for 'Platelets', 'RBC', 'WBC', respectively.

```python
train:/content/bccd_rbc_wbc_platelets-1/train/images
val: /content/bccd_rbc_wbc_platelets-1/valid/images
test: /content/bccd_rbc_wbc_platelets-1/test/images

nc: 3
names: ['Platelets', 'RBC', 'WBC']

roboflow:
  workspace: research-plgpi
  project: bcdd_rbc_wbc_platelets
  version: 1
  license: MIT
  url: https://universe.roboflow.com/research-plgpi/bcdd_rbc_wbc_platelets/dataset/1
```

# Load yolo11n.pt (pretrained model)

```python
# copy code from https://docs.ultralytics.com/tasks/detect/#faq

# Check the directory at /content/ and see that the dataset directory is already prepared

from ultralytics import YOLO

# Download a pretrained model (largest model) in "/content/" directory i.e. local folder
model = YOLO("yolo11n.pt")
```



# Selct GPU

```python
from ultralytics import YOLO
import os

print("torch.cuda.is_available():", __import__('torch').cuda.is_available())
print("torch.cuda.device_count():", __import__('torch').cuda.device_count())

device_to_use = 0 if __import__('torch').cuda.is_available() else 'cpu'
```

# Fine tune yolo11n.pt wuith above data

Modiy **data.yaml** as per train, valid and test directory locations

```python
model = YOLO("/content/yolo11n.pt")  # Load pretrained yolo11n model

results = model.train(data="/content/bccd_rbc_wbc_platelets-1/data.yaml", epochs=200, imgsz=640)
```
The _model.train()_ uses default hyper parameters that are not mentiond above as per ultralytics see [https://docs.ultralytics.com/modes/train/#train-settings]https://docs.ultralytics.com/modes/train/#train-settings

The result finetuned model is saved as: **/content/runs/detect/train/weights/best.pt** 

The name of model is **best.pt**

Downlaod **best.pt** at local machine.

# Validation of the finetune model

# First run the code to get the validation metrics of best.pt


```python
from ultralytics import YOLO

model_path = "/content/runs/detect/train/weights/best.pt"  # your trained model
data_yaml_path = "/content/bccd_rbc_wbc_platelets-1/data.yaml"  # dataset configuration file

model = YOLO(model_path)
metrics = model.val(data=data_yaml_path)  # ensure data.yaml points to the correct valid set

# Extract overall metrics
overall_precision = metrics.box.mp       # mean precision over all classes
overall_recall = metrics.box.mr          # mean recall over all classes
overall_map50 = metrics.box.map50        # mean AP at IoU=0.5 over all classes
overall_map = metrics.box.map            # mean AP at IoU=0.5:0.95 over all classes
overall_map75 = metrics.box.map75        # mean AP at IoU=0.75 over all classes

# Extract per-class metrics
class_names = model.names  # or load from data.yaml if needed, same as model.names
class_metrics = []
for i, cname in enumerate(class_names):
    p, r, ap50, ap = metrics.box.class_result(i)
    class_metrics.append((cname, p, r, ap50, ap))

print("Overall Metrics:")
print(f"Precision: {overall_precision}")
print(f"Recall: {overall_recall}")
print(f"mAP50: {overall_map50}")
print(f"mAP50-95: {overall_map}")
print(f"mAP75: {overall_map75}")
print("\nPer-Class Metrics:")
for (cname, p, r, ap50, ap) in class_metrics:
    print(f"{cname}: Precision={p}, Recall={r}, mAP50={ap50}, mAP50-95={ap}")
```

The output might be as below:

```python
Ultralytics 8.3.49 🚀 Python-3.10.12 torch-2.5.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
YOLO11n summary (fused): 238 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
val: Scanning /content/bccd_rbc_wbc_platelets-1/valid/labels.cache... 73 images, 0 backgrounds, 0 corrupt: 100%|██████████| 73/73 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 5/5 [00:02<00:00,  2.15it/s]
                   all         73       1003        0.9      0.915      0.948      0.653
             Platelets         43         73      0.882      0.959      0.961      0.502
                   RBC         70        855       0.82      0.787      0.888       0.63
                   WBC         73         75      0.998          1      0.995      0.826
Speed: 0.2ms preprocess, 3.8ms inference, 0.0ms loss, 2.3ms postprocess per image
Results saved to runs/detect/val3
Overall Metrics:
Precision: 0.8998657967724281
Recall: 0.9152413015416975
mAP50: 0.9482967626275897
mAP50-95: 0.6529025986330599
mAP75: 0.7199225312247104

Per-Class Metrics:
0: Precision=0.8820047185253768, Recall=0.958904109589041, mAP50=0.961433378998409, mAP50-95=0.5023530432704303
1: Precision=0.8196728808767741, Recall=0.7868197950360514, mAP50=0.8884569088843599, mAP50-95=0.6302822447945686
2: Precision=0.9979197909151334, Recall=1.0, mAP50=0.995, mAP50-95=0.8260725078341811
```

# Hard code above metrics into below code 'app.py' respective metrics

**app.py**

```python
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import yaml
import gradio as gr
import pandas as pd

model_path = "best.pt"
data_yaml_path = "data.yaml"

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
```
# Prepare requirements.txt

**requirements.txt**

```python
ultralytics 
gradio
```

# Create huggingface account and space (free)

Upload all files to the Huggingface Space with a suitable name such as "csepartha/yolon11_bccd_objection_detection"

![image](https://github.com/user-attachments/assets/4805b4b6-30b6-4599-95a6-5ba5ea086fdb)


# Run app 

Huggingface space: https://huggingface.co/spaces/csepartha/yolon11_bccd_objection_detection

Upload a test iamge from BCCD local folder to the app and click **Run Inference** below output along with Overall class and individual cell centric comaprison table is shown.

![image](https://github.com/user-attachments/assets/f276fb39-556e-4603-8af2-b951c5e6eb90)
![image](https://github.com/user-attachments/assets/827fc5a0-a7bb-4798-907a-fd4e94dfd4f5)

