# Pneumonia Detection from Chest X-rays using DenseNet121

This project implements a deep learning model to classify chest X-ray images into 'NORMAL' or 'PNEUMONIA' categories. It utilizes a pre-trained DenseNet121 model, fine-tuned on the Chest X-Ray Images (Pneumonia) dataset.

## Overview

The primary goal is to build an accurate classifier for pneumonia detection using chest radiography. The code performs the following key steps:

1.  **Data Loading & Preprocessing:** Loads images from the specified dataset path and applies transformations (resizing, rotation, flipping, normalization, grayscale conversion) for data augmentation and standardization.
2.  **Dataset Splitting:** Splits the training data into training and validation sets.
3.  **Class Imbalance Handling:** Calculates and applies class weights to the loss function to address the imbalance between 'NORMAL' and 'PNEUMONIA' samples.
4.  **Model Architecture:** Uses a DenseNet121 model pre-trained on ImageNet. The last few layers are unfrozen for fine-tuning, and the final classifier layer is replaced to match the number of classes in our dataset (2).
5.  **Training:** Trains the model using Adam optimizer, Cross-Entropy Loss (with class weights), and a learning rate scheduler. Includes early stopping to prevent overfitting based on validation loss.
6.  **Evaluation:** Evaluates the trained model on the test set using metrics like classification report and confusion matrix.
7.  **Visualization:** Plots training and validation loss/accuracy curves.
8.  **Inference:** Provides a function to test the model on single, unseen images.

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset, commonly found on platforms like Kaggle.

* **Source (Example):** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* The code expects the dataset to be structured as follows:
    ```
    <dataset_root_directory>/
    ├── train/
    │   ├── NORMAL/
    │   │   └── *.jpeg
    │   └── PNEUMONIA/
    │       └── *.jpeg
    └── test/
        ├── NORMAL/
        │   └── *.jpeg
        └── PNEUMONIA/
            └── *.jpeg
    ```
* Update the `DATASET_PATH` variable in the script to point to your `<dataset_root_directory>`.


![Screenshot (419)](https://github.com/user-attachments/assets/7e9b23c4-1c70-42fc-9b3d-3dcf7dac5d29)
![Screenshot (420)](https://github.com/user-attachments/assets/27a555b0-1509-46cc-8bcb-56975ddcd36f)
![Screenshot (421)](https://github.com/user-attachments/assets/6c7707c9-43f8-46ae-86c1-dec134e8331a)
![Screenshot (422)](https://github.com/user-attachments/assets/629934f6-8fa7-4215-9df9-4cd8c5984ee1)



