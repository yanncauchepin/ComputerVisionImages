import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the path to your dataset directory
dataset_dir = f"{os.getcwd()}/Dataset"

# Define the class labels
class_labels = ['no', 'yes']

# Initialize empty lists to store images and corresponding labels
images = []
labels = []

# Define the target size for resizing
target_size = (600, 600)  # Adjust as needed

# Iterate through each class label
for label in class_labels:
    # Create the path to the class label directory
    class_dir = os.path.join(dataset_dir, label)

    # Iterate through each image file in the class label directory
    for image_file in os.listdir(class_dir):
        # Create the path to the image file
        image_path = os.path.join(class_dir, image_file)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # If the image is successfully loaded
        if image is not None:
            # Resize the image to the target size
            image = cv2.resize(image, target_size)

            # Convert the image to grayscale (optional)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Flatten the image into a 1D array
            image = image.flatten()

            # Append the preprocessed image and corresponding label to the lists
            images.append(image)
            labels.append(label)

# Convert the lists to NumPy arrays for further processing
images = np.array(images)
labels = np.array(labels)

# Perform train/validation/test split
train_images, test_images, train_labels, test_labels = \
    train_test_split(images, labels, test_size=0.2, random_state=1)

# Initialize and train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(train_images, train_labels)

# Perform predictions on the test set
predictions = classifier.predict(test_images)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
