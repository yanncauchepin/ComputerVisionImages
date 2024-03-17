import numpy as np

import samples
import samples.classifier_brain_tumor.preprocessing as BrainTumorPreprocessing

import cv2

from tensorflow import keras
from tensorflow import image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class ClassifierBrainTumor():
    
    def __init__(self):
        self.df_brain_tumor = BrainTumorPreprocessing.load_dataframe()
        
    def __run_cnn_tensorflow(self):
        
        images = self.df_brain_tumor['images']
        for idx in range(len(images)):
            images[idx] = image.resize(images[idx], [600,600])
            images[idx] = images[idx]/255.0
        images = np.array(images)
        labels = np.array(self.df_brain_tumor['labels'])
        labels = np.where(labels == 'yes', 1., 0.)
        
        train_images, test_images, train_labels, test_labels = \
            train_test_split(images, labels, test_size=0.2, random_state=42)

        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(600, 600, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

        test_loss, test_accuracy = model.evaluate(test_images, test_labels)
        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_accuracy)
        
    def __run_cnn_pytorch(self):
        
        images = self.df_brain_tumor['images']
        resized_images = []
        
        for idx in range(len(images)):
            resized_image = cv2.resize(images[idx], (600, 600))
            resized_image = resized_image / 255.0
            resized_images.append(resized_image)

        images = np.array(resized_images)
        labels = np.array(self.df_brain_tumor['labels'])
        labels = np.where(labels == 'yes', 1, 0)

        train_images, test_images, train_labels, test_labels = \
            train_test_split(images, labels, test_size=0.2, random_state=42)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = CustomDataset(train_images, train_labels, transform=transform)
        test_dataset = CustomDataset(test_images, test_labels, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        nb_epoch = 10
        for epoch in range(nb_epoch):
            model.train()
            with tqdm(train_loader, desc=f'Epoch {epoch +1}/{nb_epoch}', unit='batch') as train_loader:
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
            model.eval()
            with tqdm(test_loader, desc='Evaluation', unit='batch') as test_loader:
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loader.set_postfix(accuracy=correct/total)

        accuracy = correct / total
        print('Test Accuracy:', accuracy)
    
    def __run_sklearn(self, algorithm):
        
        images = self.df_brain_tumor['images']
        for idx in range(len(images)):
            images[idx] = cv2.resize(image[idx], (600,600))
            images[idx] = images[idx].flatten()
        images = np.array(images)
        labels = np.array(self.df_brain_tumor['labels'])
        
        train_images, test_images, train_labels, test_labels = \
            train_test_split(images, labels, test_size=0.2, random_state=1)

        if algorithm == 'logistic_regression':
            classifier = LogisticRegression()
            classifier.fit(train_images, train_labels)
            predictions = classifier.predict(test_images)
        elif algortihm == 'support_vector_machine':
            classifier = SVC()
            classifier.fit(train_images, train_labels)
            predictions = classifier.predict(test_images)
        elif algorithm == 'random_forest':
            classifier = RandomForestClassifier()
            classifier.fit(train_images, train_labels)
            predictions = classifier.predict(test_images)
        elif algortihm == 'k_neighbors':
            classifier = KNeighborsClassifier()
            classifier.fit(train_images, train_labels)
            predictions = classifier.predict(test_images)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy : {accuracy}")
        
    def run(self, algorithm):
        if algorithm == 'cnn_tensorflow':
            self.__run_cnn_tensorflow()
        elif algorithm == 'cnn_pytorch':
            self.__run_cnn_pytorch()
        elif algorithm == 'lr_sklearn':
            self.__run_sklearn("logistic_regression")
        elif algorithm == 'svm_sklearn':
            self.__run_sklearn("support_vector_machine")
        elif algorithm == 'rf_sklearn':
            self.__run_sklearn("random_forest")
        elif algorithm == 'kn_sklearn':
            self.__run_sklearn("k_neighbors")
        else:
            raise Exception(f'Algorithm {algorithm} not recognized.')
        

if __name__ == '__main__':
    classifierbraintumor = ClassifierBrainTumor()
    classifierbraintumor.run('cnn_pytorch')