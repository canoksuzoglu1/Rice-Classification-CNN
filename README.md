# Rice-Classification-CNN

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) model using **TensorFlow** to classify images of different rice varieties. The project uses a publicly available dataset containing images of rice grains and covers key steps such as dataset loading, preprocessing, data augmentation, model building, training, and model evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
   - [1. Libraries and Constants](#1-libraries-and-constants)
   - [2. Data Loading and Sampling](#2-data-loading-and-sampling)
   - [3. Image Preview](#3-image-preview)
   - [4. Dataset Splitting](#4-dataset-splitting)
   - [5. Data Preprocessing](#5-data-preprocessing)
   - [6. Model Definition](#6-model-definition)
   - [7. Model Compilation and Training](#7-model-compilation-and-training)
   - [8. Model Evaluation and Prediction](#8-model-evaluation-and-prediction)
   - [9. Model Saving and Loading](#9-model-saving-and-loading)
   - [Key Points](#key-points)
- [Results](#results)
- [License](#license)

## Project Overview

This repository contains a TensorFlow-based image classification project where we train a CNN model to classify different varieties of rice grains. The goal is to accurately classify images into one of five rice varieties (Arborio, Basmati, Ipsala, Jasmine, and Karacadag). The project includes functionalities for data augmentation, dataset splitting, and model checkpointing to save the best performing model.

## Dataset Information

**Rice Image Dataset**  
Link: [Kaggle - Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  
Original Dataset Link: [Murat Koklu Dataset](https://www.muratkoklu.com/datasets/)

### Citation Request
If you use this dataset, please cite the following articles:

- Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. *Computers and Electronics in Agriculture, 187*, 106285. [DOI:10.1016/j.compag.2021.106285](https://doi.org/10.1016/j.compag.2021.106285)
  
- Cinar, I., & Koklu, M. (2021). Determination of Effective and Specific Physical Features of Rice Varieties by Computer Vision In Exterior Quality Inspection. *Selcuk Journal of Agriculture and Food Sciences, 35(3)*, 229-243. [DOI:10.15316/SJAFS.2021.252](https://doi.org/10.15316/SJAFS.2021.252)
  
- Cinar, I., & Koklu, M. (2022). Identification of Rice Varieties Using Machine Learning Algorithms. *Journal of Agricultural Sciences*. [DOI:10.15832/ankutbd.862482](https://doi.org/10.15832/ankutbd.862482)
  
- Cinar, I., & Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. *International Journal of Intelligent Systems and Applications in Engineering, 7(3)*, 188-194. [DOI:10.18201/ijisae.2019355381](https://doi.org/10.18201/ijisae.2019355381)

### Highlights

- The dataset contains images from five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.
- The dataset includes 75,000 images, with 15,000 images for each rice variety.
- Artificial Neural Network (ANN), Deep Neural Network (DNN), and Convolutional Neural Network (CNN) models were used for classification.
- The CNN model achieved a 100% classification accuracy for rice varieties.

## Prerequisites

Before running the code, you will need to install the following dependencies:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install all required libraries using `pip`:

```bash
pip install tensorflow numpy matplotlib
```

## Dataset Structure
The project assumes that you have an image dataset organized in the following directory structure:

- **Data/**  
  - **Arborio/**  
    - `image_1.jpg`  
    - `image_2.jpg`  
    - `...`  
  - **Basmati/**  
    - `image_1.jpg`  
    - `image_2.jpg`  
    - `...`  
  - **Ipsala/**  
    - `image_1.jpg`  
    - `image_2.jpg`  
    - `...`  
  - **Jasmine/**  
    - `image_1.jpg`  
    - `image_2.jpg`  
    - `...`  
  - **Karacadag/**  
    - `image_1.jpg`  
    - `image_2.jpg`  
    - `...`  


Each subdirectory under `/Data` corresponds to one of the five rice varieties, and the images within the subdirectories belong to that variety.

## Installation
1. Clone the repository:
bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2. Install the required dependencies:
bash
pip install -r requirements.txt

3. Place your dataset in the Data/ directory as shown above.

## Usage

### 1. Libraries and Constants
- Import necessary libraries like TensorFlow, Keras, and Matplotlib.
- Set constants like `IMAGE_SIZE`, `BATCH_SIZE`, `CHANNELS`, and `EPOCHS` to configure the model's input size and training parameters.

```python
IMAGE_SIZE = 250
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
```
### 2. Data Loading and Sampling
- Load the dataset using image_dataset_from_directory with specified image_size and batch_size.
- Randomly sample the dataset using a custom sampling function to reduce its size for quicker iteration during development.

```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

dataset = sample_dataset(dataset, fraction=1/15)
```

### 3. Image Preview
Visualize a few images from the dataset to ensure loading and pre-processing are correct.
```python
plt.figure(figsize=(10, 10))
for image_batch, image_label in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
```
### 4. Dataset Splitting
Use a custom function to split the dataset into training, validation, and test sets as TensorFlow does not have a built-in function for this.
```python
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
```
### 5. Data Preprocessing
Cache, shuffle, and prefetch the data for better training efficiency.
Apply image resizing and scaling transformations using `layers.Rescaling` and `layers.Resizing`.
```python
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/256)
])
```
### 6. Model Definition
- A CNN model is built with two Conv2D layers followed by pooling, and a Dense layer for classification.
- Apply data augmentation to improve the model's robustness.
```python
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])
```
### 7. Model Compilation and Training
- Compile the model using Adam optimizer with a reduced learning rate (`learning_rate=1e-5`) for better convergence.
- Use `ModelCheckpoint` to save the best version of the model based on validation loss.
```python
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_callback]
)
```
### 8. Model Evaluation and Prediction
- After training, evaluate the model using the test dataset.
- Use a prediction function to classify new images and visualize the results with confidence scores.
```python
score = model.evaluate(test_ds)
```
### 9. Model Saving and Loading
- Save and load the trained model for future inference.
```python
model.save(f"./models/{model_version}.keras")
model = tf.keras.models.load_model(model_path)
```
### Key Points:
- Data Preprocessing: Ensure images are correctly resized and rescaled.
- Model Architecture: Keep the architecture simple with a small number of layers, especially for quicker prototyping.
- Checkpointing: Always save the best model during training to avoid overfitting.

## Results
The model can be evaluated by visualizing predictions on test images. The script includes code to display predicted labels and confidence scores along with the actual labels of test images.

Example:
Predicted: Arborio (95.12%)
Actual: Arborio
