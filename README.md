# Lung_Infection_Detection_Using_CNN


# Image Classification Project with CNNs and Transfer Learning

This project implements a multi-class image classification model using Convolutional Neural Networks (CNNs) and Transfer Learning with TensorFlow and Keras. The objective is to classify images into three classes: `Healthy`, `Type1disease`, and `Type2disease`. This project follows a series of model-building steps, starting from a basic CNN model, adding data augmentation, and then using transfer learning with pre-trained models (MobileNetV2 and DenseNet121).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Building and Training](#2-model-building-and-training)
  - [3. Transfer Learning with MobileNetV2 and DenseNet121](#3-transfer-learning-with-mobilenetv2-and-densenet121)
- [Evaluation and Results](#evaluation-and-results)
- [Visualization of Training Results](#visualization-of-training-results)
- [License](#license)

---

## Overview

The main goal of this project is to train a model that can accurately classify medical images into three classes, namely `Healthy`, `Type1disease`, and `Type2disease`. This project demonstrates three stages of model building:

1. A basic CNN model
2. A CNN model with data augmentation and dropout layers
3. Transfer learning with MobileNetV2 and DenseNet121

Each model's accuracy and performance on the test set are evaluated to identify the most effective model for this classification task.

## Dataset

The dataset consists of images divided into training, validation, and test directories, each containing three classes:
- `Healthy`
- `Type1disease`
- `Type2disease`

### Directory Structure

```
data/
├── train/
│   ├── Healthy/
│   ├── Type1disease/
│   └── Type2disease/
└── test/
    ├── Healthy/
    ├── Type1disease/
    └── Type2disease/
```

### Image Distribution
The dataset contains 251 training images and 66 test images.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

### Install required packages

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Vaibhav99mishra/Lung_Infection_Detection_Using_CNN.git
   cd Lung_Infection_Detection_Using_CNN
   ```

2. Set up the dataset directory structure as described above.

3. Run each part of the project sequentially as described below.

## Project Workflow

### 1. Data Preprocessing

1. **Data Loading**: Images are loaded from training and test directories using `tf.keras.utils.image_dataset_from_directory`.
2. **Visualization**: Displays sample images from each class for visual inspection.
3. **Dataset Preparation**: Creates training, validation, and test datasets with buffered prefetching and caching for efficient loading during training.

### 2. Model Building and Training

#### Model 1: Basic CNN Model

- **Architecture**: A simple CNN model with three convolutional layers followed by max pooling, flatten, and dense layers.
- **Compilation**: The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss.
- **Training**: The model is trained for 10 epochs, and performance is evaluated on the test set.

#### Model 2: CNN with Data Augmentation and Dropout

- **Data Augmentation**: Includes random horizontal flipping, rotation, and zoom to improve robustness.
- **Architecture**: The CNN model from Model 1 is expanded with a dropout layer to reduce overfitting.
- **Training**: The model is trained with data augmentation, resulting in a more generalized model.

### 3. Transfer Learning with MobileNetV2 and DenseNet121

#### Model 3: MobileNetV2

- **Transfer Learning**: MobileNetV2 is used as a base model with pre-trained weights from ImageNet. Only the top layers are customized for the current classification task.
- **Architecture**: Includes global average pooling, dropout, and batch normalization layers on top of MobileNetV2.
- **Training**: The model is trained for 10 epochs with early stopping, and performance is evaluated on the test set.

#### Model 4: DenseNet121

- **Transfer Learning**: DenseNet121 is used as a base model with pre-trained weights from ImageNet.
- **Architecture**: Similar to Model 3, with additional layers for dropout and batch normalization.
- **Training**: The model is trained for 15 epochs with early stopping to prevent overfitting.

## Evaluation and Results

Each model is evaluated using a classification report that includes precision, recall, and f1-score for each class. The overall accuracy is calculated, allowing for a comparison between the models.

### Example Output

```
Color Model Classification Report:
              precision    recall  f1-score   support

           0       0.37      0.35      0.36        20
           1       0.50      0.46      0.48        26
           2       0.30      0.35      0.33        20

    accuracy                           0.39        66
   macro avg       0.39      0.39      0.39        66
weighted avg       0.40      0.39      0.40        66
```

## Visualization of Training Results

Training and validation accuracy and loss are plotted for each model to visualize the model's performance over epochs.

```python
# Sample code for plotting training and validation accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

## License

This project is licensed under the MIT License.
