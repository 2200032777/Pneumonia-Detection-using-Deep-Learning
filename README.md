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








##code

model.py:-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import densenet121
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter
from PIL import Image

# ✅ Step 1: Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Step 2: Define Dataset Paths
DATASET_PATH = "../input/chest-xray-pneumonia/chest_xray/chest_xray"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Dataset directories not found. Please check the paths.")

# ✅ Step 3: Define Transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Step 4: Load Datasets
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

# Class mapping
print("Class to Index Mapping:", full_train_dataset.class_to_idx)

# ✅ Step 5: Train/Validation Split
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ Step 6: Handle Class Imbalance
targets = [label for _, label in train_dataset]
class_counts = Counter(targets)
class_weights = torch.tensor([1.0 / class_counts[i] for i in range(len(class_counts))], device=device)
print("Class Weights:", class_weights)

# ✅ Step 7: Load & Modify DenseNet121
model = densenet121(pretrained=True)

# Unfreeze last 10 layers
for param in model.features[-10:].parameters():
    param.requires_grad = True

# Replace classifier
num_classes = len(full_train_dataset.classes)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# ✅ Step 8: Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ✅ Step 9: Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=3):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0.0, 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                class_correct[labels[i]] += (predicted[i] == labels[i]).item()
                class_total[labels[i]] += 1

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / len(train_loader.dataset))
        class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
        print(f"Epoch {epoch+1}: Train Loss={train_loss[-1]:.4f}, Train Acc={train_acc[-1]:.4f}, Class Acc={class_accuracy}")

        # Validation
        model.eval()
        val_running_loss, val_correct = 0.0, 0
        val_class_correct = [0] * num_classes
        val_class_total = [0] * num_classes
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                _, predicted = torch.max(outputs, 1)
                for i in range(len(labels)):
                    val_class_correct[labels[i]] += (predicted[i] == labels[i]).item()
                    val_class_total[labels[i]] += 1

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(val_correct / len(val_loader.dataset))
        val_class_accuracy = [val_class_correct[i] / val_class_total[i] if val_class_total[i] > 0 else 0 for i in range(num_classes)]
        print(f"Epoch {epoch+1}: Val Loss={val_loss[-1]:.4f}, Val Acc={val_acc[-1]:.4f}, Val Class Acc={val_class_accuracy}")

        scheduler.step()

        # Early Stopping Check
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("⛔️ Early stopping triggered!")
                break

    return train_loss, train_acc, val_loss, val_acc

# ✅ Step 10: Train the Model
train_loss, train_acc, val_loss, val_acc = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
)

# ✅ Step 11: Save the Model
torch.save({
    'epoch': 20,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss[-1],
}, "pneumonia_densenet121.pth")
print("✅ Model saved successfully")

# ✅ Step 12: Test & Evaluate Model
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=full_train_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

test_model(model, test_loader)

# ✅ Step 13: Visualize Training Results
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# ✅ Step 14: Test on Individual Samples
def test_single_image(image_path, model, transform, class_mapping):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_mapping[predicted.item()]
    return predicted_class

# Test example (Update your paths)
pneumonia_image_path = "path_to_pneumonia_image.jpg"
normal_image_path = "path_to_normal_image.jpg"

print("Prediction for Pneumonia Image:", test_single_image(pneumonia_image_path, model, transform, full_train_dataset.class_to_idx))
print("Prediction for Normal Image:", test_single_image(normal_image_path, model, transform, full_train_dataset.class_to_idx))
