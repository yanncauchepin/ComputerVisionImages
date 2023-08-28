import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the path to your dataset directory
dataset_dir = f"{os.getcwd()}/Dataset"

# Define the class labels
class_labels = ['no', 'yes']

# Initialize empty lists to store images and corresponding labels
images = []
labels = []

# Iterate through each class label
for label in class_labels:
    # Create the path to the class label directory
    class_dir = os.path.join(dataset_dir, label)

    # Iterate through each image file in the class label directory
    for image_file in os.listdir(class_dir):
        # Create the path to the image file
        image_path = os.path.join(class_dir, image_file)

        # Load the image using TensorFlow
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(600, 600))

        # Convert the image to a NumPy array
        image_array = tf.keras.preprocessing.image.img_to_array(image)

        # Preprocess the image (e.g., normalize pixel values)
        image_array = image_array / 255.0

        # Append the preprocessed image and corresponding label to the lists
        images.append(image_array)
        labels.append(label)

# Convert the lists to NumPy arrays for further processing
images = np.array(images)
labels = np.array(labels)
labels = np.where(labels == 'yes', 1., 0.)

# Perform train/validation/test split
train_images, test_images, train_labels, test_labels = \
    train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model architecture
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

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
