# Moroccan Date Fruit Recognition Using Transfer Learning and Texture-Aware Color Attention

Project Title : Moroccan Date Fruit Recognition Using Transfer Learning and Texture-Aware Color Attention

# Overview
This project focuses on the automatic classification of Moroccan date fruit varieties using deep learning.
It leverages transfer learning and fine-tuning of VGG19, with a custom-designed texture-color attention mechanism based on HSV color space and Gabor filters.
The main goal is to improve recognition accuracy by emphasizing the visual cues that matter most in this domain: color and texture informations.
While size and shape are generally useful features in date fruit classification, they are less reliable in this study.
The dataset was collected in uncontrolled conditions, where: Camera angles vary significantly and the  distances between the fruit and camera are inconsistent

As a result, spatial information such as scale or contour shape becomes highly variable and not meaningful for training.
Thus, this work focuses on more robust visual cues: color and texture.

# Objective
Classify individual Moroccan date fruits into their correct variety.
Handle real-world conditions with semi-controlled lighting and background.
Explore how custom preprocessing (HSV + Gabor) can enhance feature learning.

# Why This Approach?
Most pretrained CNNs are tuned for general object recognition.
However, in date fruit classification, the key discriminators are:
Color (shades of brown, red , golden, yellow) and Texture (smooth vs. wrinkled, glossy vs. matte , rough vs soft)

To shift the model’s attention toward these features, the input images are:
Converted to HSV color space → separates color (Hue) from intensity
Filtered using Gabor filters → highlight texture patterns

These components are stacked into a custom 3-channel input:
[Hue, Saturation, Gabor Response] Which is then fed into VGG19 for feature extraction and fine-tuning

# Dataset
Contains images of individual Moroccan date fruits.
Captured in semi-uncontrolled environments.
Each image contains a single fruit, placed on simple backgrounds.
Number of classes: 9 varieties.

The data/ directory included in this repository contains only one sample image  per class for demonstration purposes.
The full dataset used in training and evaluation is not published here due to size and storage limitations.

If you'd like to test the pipeline on your own dataset, you can organize your images in the same folder-based format:

data/

├── Almajhoul/

│   ├── image1.jpg

│   └── image2.jpg

├── Bofgous/

│   ├── image1.jpg

│   └── image2.jpg

...

and update the code to match your number of classes


# Technical Details
Model: VGG19 pretrained on ImageNet

Attention Mechanism: Color-Texture Enhanced Input (CTEI) via HSV + Gabor

Input size: 224×224

Preprocessing: HSV conversion , Gabor filtering (theta = 0, kernel size 21×21)

Fine-Tuning Strategy: Used only the first 16 layers of VGG19 (4 convolutional blocks) to reduce the number of parameters and focus on generalizable features.
First 13 layers were frozen, allowing the last 3 layers to adapt to task-specific patterns (color and texture).

Classification head includes: GlobalAveragePooling2D , Dense(32, activation='relu', kernel_regularizer=L2(0.001)) , Dropout(0.3) , Dense(num_classes, activation='softmax')

# Evaluation Protocol
Dataset split: 70% training, 15% validation, 15% testing

Evaluation metric: test set accuracy

# Best Accuracy:  94.81%

loss evolution :
<img width="381" height="316" alt="Sans titre" src="https://github.com/user-attachments/assets/fbb31285-0a04-40de-bffd-3b348d1f9954" />

accuracy evolution : 
<img width="381" height="316" alt="Sans titre-1" src="https://github.com/user-attachments/assets/2970b18d-7ac3-4fa4-aa16-d6040afaf1ae" />



