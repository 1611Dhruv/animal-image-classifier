import os
import random
import shutil
import zipfile

import kaggle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


# Step 1: Download the dataset using Kaggle API
def download_dataset():
    dataset = "iamsouravbanerjee/animal-image-dataset-90-different-animals"
    kaggle.api.dataset_download_files(dataset, path="data", unzip=True)


# Step 2: Split the dataset into training and testing sets
def split_dataset(dataset_dir, train_ratio=0.8):
    classes = os.listdir(dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        train_images, test_images = train_test_split(
            images, train_size=train_ratio, random_state=42
        )

        train_class_dir = os.path.join(train_dir, cls)
        test_class_dir = os.path.join(test_dir, cls)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img in train_images:
            shutil.move(
                os.path.join(class_dir, img), os.path.join(train_class_dir, img)
            )

        for img in test_images:
            shutil.move(os.path.join(class_dir, img), os.path.join(test_class_dir, img))

        shutil.rmtree(class_dir)


# Step 3: Conduct basic analysis using Matplotlib and Seaborn
def analyze_dataset(train_dir, test_dir):
    train_data = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    test_data = datasets.ImageFolder(test_dir, transform=transforms.ToTensor())

    train_classes = [train_data.classes[idx] for _, idx in train_data]
    test_classes = [test_data.classes[idx] for _, idx in test_data]

    train_df = pd.DataFrame(train_classes, columns=["class"])
    test_df = pd.DataFrame(test_classes, columns=["class"])

    plt.figure(figsize=(16, 6))
    sns.countplot(data=train_df, x="class")
    plt.xticks(rotation=90)
    plt.title("Class Distribution in Training Set")
    plt.show()

    plt.figure(figsize=(16, 6))
    sns.countplot(data=test_df, x="class")
    plt.xticks(rotation=90)
    plt.title("Class Distribution in Testing Set")
    plt.show()


# Main function to execute the steps
def main():
    download_dataset()

    dataset_dir = "data/Animal Image Dataset/90 Animal Image Data"
    split_dataset(dataset_dir)

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    analyze_dataset(train_dir, test_dir)


if __name__ == "__main__":
    main()
