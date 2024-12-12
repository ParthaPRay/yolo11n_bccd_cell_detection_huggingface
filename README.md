# yolon11_bccd_cell_detection
This repo contains codes to fine tune yolon11 and with bccd dataset for RBC, WBC and Platelets cell object detection

# Yolon11 Fine Tuning for BCCD Dataset object detection

# Dataset

https://github.com/Shenggan/BCCD_Dataset

BCCD Dataset is a small-scale dataset for blood cells detection.

BCCD has three kind of labels :

- RBC
- WBC
- Platelets

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



![image](https://github.com/user-attachments/assets/8963ce8b-8fb4-4611-8ae7-3055080c6cb6)


