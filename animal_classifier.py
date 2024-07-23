import os

import kaggle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm  # Import tqdm for progress bar


# Custom AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Step 1: Download the dataset using Kaggle API
def download_dataset():
    dataset = "iamsouravbanerjee/animal-image-dataset-90-different-animals"
    kaggle.api.dataset_download_files(dataset, path="data", unzip=True)


# Step 2: Load datasets and split into training and testing sets
def load_and_split_dataset(base_dir, train_ratio=0.8):
    animal_dir = os.path.join(base_dir, "animals", "animals")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(animal_dir, transform=transform)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset, dataset.classes


# Step 3: Train the model with progress bar
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            dataloader = dataloaders[phase]

            # Use tqdm to create a progress bar
            for inputs, labels in tqdm(
                dataloader, desc=f"{phase.capitalize()} Phase", leave=False
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model


# Step 4: Evaluate the model
def evaluate_model(model, dataloaders, classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["test"], desc="Evaluating Model"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=classes))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")


# Step 5: Visualize CNN layers and weights
def visualize_layers_and_weights(model):
    def plot_kernels(tensor, num_cols=6):
        num_kernels = tensor.shape[0]
        num_rows = num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols, num_rows))
        for i in range(num_kernels):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.imshow(tensor[i][0, :, :].detach().cpu().numpy(), cmap="gray")
            ax.axis("off")
        plt.show()

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(f"Layer: {name}")
            plot_kernels(layer.weight)


# Main function
def main():
    if not os.path.exists("data"):
        download_dataset()

    base_dir = "data"
    train_dataset, test_dataset, classes = load_and_split_dataset(base_dir)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        "test": DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4),
    }

    model = AlexNet(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

    evaluate_model(model, dataloaders, classes)

    torch.save(model.state_dict(), "alexnet_animal_classifier.pth")
    print("Model saved as alexnet_animal_classifier.pth")

    visualize_layers_and_weights(model)


if __name__ == "__main__":
    main()
