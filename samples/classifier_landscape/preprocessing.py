import os
import cv2

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionImages/classifier_landscape/"

def load_dataframe() :
    
    images = []
    labels = []

    for set_ in ['Testing Data', 'Training Data', 'Validation Data']:
        for label in os.listdir(os.path.join(root_path, set_)):
            label_dir = os.path.join(root_path, set_, label)
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(str(label).lower())

    return {"images" : images, "labels" : labels}


if __name__ == '__main__' :

    """EXAMPLE"""

    df_landscape = load_dataframe()