import os
import cv2
import imageio

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionFaces/yales_faces/"

def load_dataframe() :

    labels = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal',
              'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
    images = []
    labels = []
    dataset = os.path.join(root_path, 'data')
    for image_file in os.listdir(dataset):
        image_path = os.path.join(dataset, image_file)
        gif_image = imageio.imread(image_path)
        image = cv2.cvtColor(gif_image, cv2.COLOR_RGB2BGR)
        if image is not None:
            images.append(image)
            labels.append(str(image_file).split('.')[-1])

    return {"images" : images, "labels" : labels}


if __name__ == '__main__' :

    """EXAMPLE"""

    df_yales_faces = load_dataframe()
