import os
import cv2
import numpy as np

class RandomImage():
    
    def __init__(self):
        pass

    @staticmethod
    def random_image(root_path, width, height, type_image='bgr'):
        if type_image=='bgr':
            image_array = np.random.randint(0, 256, width*height*3)
            # alternative = np.array(os.urandom(width*height*3))
            image = image_array.reshape(width, height, 3)
            cv2.imwrite(os.path.join(root_path, 'random_bgr_image.png'), image)
        elif type_image=='gray':
            image_array = np.random.randint(0, 256, width*height)
            # alternative = np.array(os.urandom(width*height))
            image = image_array.reshape(width, height)
            cv2.imwrite(os.path.join(root_path, 'random_gray_image.png'), image)
        else:
            raise Exception(f'Type image {type_image} not recognized.')
            
        return image