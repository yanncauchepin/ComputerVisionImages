import os
import cv2

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionImages/classifier_brain_tumor/"

def load_dataframe() :

    class_labels = ['no', 'yes']
    
    images = []
    labels = []

    for label in class_labels:
        class_dir = os.path.join(root_path, label)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(label)

    return {"images" : images, "labels" : labels}


if __name__ == '__main__' :

    """EXAMPLE"""

    df_brain_tumors = load_dataframe()