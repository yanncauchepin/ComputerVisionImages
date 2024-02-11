# -*- coding: utf-8 -*-
"""
@author: yanncauchepin
"""

import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, class_labels, transform=None):
        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []

        for label in self.class_labels:
            class_dir = os.path.join(self.dataset_dir, label)
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                images.append(image_path)
                labels.append(1 if label == 'yes' else 0)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


dataset_dir = f"/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionPictures/ClassifierBrainTumor"
class_labels = ['no', 'yes']

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
])

custom_dataset = CustomDataset(dataset_dir, class_labels, transform=transform)

train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 75 * 75, 128)
        self.fc2 = nn.Linear(128, len(class_labels))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 75 * 75)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

nb_epoch = 10
for epoch in range(nb_epoch):
    with tqdm(train_loader, desc=f'Epoch {epoch +1}/{nb_epoch}', unit='batch') as train_loader :
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loader.set_postfix(loss=loss.item())

correct = 0
total = 0
with torch.no_grad():
    with tqdm(test_loader, desc='Evaluation', unit='batch') as test_loader :
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader.set_postfix(accuracy=correct/total)

accuracy = correct / total
print('Test Accuracy:', accuracy)
