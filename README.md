# yolo11n_bccd_cell_detection
This repo contains codes to fine tune yolon11 and with bccd dataset for RBC, WBC and Platelets cell object detection

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
