# -*- coding: utf-8 -*-
"""
@author: yanncauchepin
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset_dir = f"/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionPictures/ClassifierBrainTumor"

class_labels = ['no', 'yes']

images = []
labels = []

target_size = (600, 600)

for label in class_labels:
    class_dir = os.path.join(dataset_dir, label)
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, target_size)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            images.append(image)
            labels.append(label)

images = np.array(images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = \
    train_test_split(images, labels, test_size=0.2, random_state=1)

print("Logistic Regression")

classifier = LogisticRegression()
classifier.fit(train_images, train_labels)

predictions = classifier.predict(test_images)

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy : {accuracy}")

print("Support Vector Machine")

classifier = SVC()
classifier.fit(train_images, train_labels)

predictions = classifier.predict(test_images)

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy : {accuracy}")

print("Random Forest Classifier")

classifier = RandomForestClassifier()
classifier.fit(train_images, train_labels)

predictions = classifier.predict(test_images)

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy : {accuracy}")

print("K Neighbors Classifier")

classifier = KNeighborsClassifier()
classifier.fit(train_images, train_labels)

predictions = classifier.predict(test_images)

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy : {accuracy}")
