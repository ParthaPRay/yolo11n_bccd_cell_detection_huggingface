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

# Roboflow project for Yolo11

Then select Roboflow project for Yolo11 as below

![image](https://github.com/user-attachments/assets/45e8136d-4d57-4d50-80c0-b7bf2d167d35)

It will allow to download the above projct by below code

![image](https://github.com/user-attachments/assets/3a4b9705-cf66-4b97-8f70-e5890a124b8a)

# Then code below


'''python
# copy code from https://docs.ultralytics.com/tasks/detect/#faq

# Check the directory at /content/ and see that the dataset directory is already prepared

from ultralytics import YOLO

# Load a pretrained model (largest model) in "/content/" directory i.e. local folder
model = YOLO("yolo11n.pt")

'''


