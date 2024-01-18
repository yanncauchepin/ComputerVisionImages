#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:36:27 2024

@author: yanncauchepin
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset_dir = f"{os.getcwd()}/Dataset"

class_labels = ['no', 'yes']

images = []
labels = []

for label in class_labels:
    class_dir = os.path.join(dataset_dir, label)
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(600, 600))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0
        images.append(image_array)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)
labels = np.where(labels == 'yes', 1., 0.)

train_images, test_images, train_labels, test_labels = \
    train_test_split(images, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(600, 600, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
