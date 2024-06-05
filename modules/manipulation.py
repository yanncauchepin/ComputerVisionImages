import os
import cv2
import numpy as np


class Manipulation():
    """
    numpy.array can be replaced by cv2.UMat
    UMat2nparray : umat.get() = nparray
    """
    
    
    def __init__(self):
        pass
    
    
    @staticmethod 
    def byte_array(image):
        return bytearray(image)
    
    
    @staticmethod
    def read_image(path, color_format='color'):
        map_ = {
            'color' : cv2.IMREAD_COLOR,
            'grayscale' : cv2.IMREAD_GRAYSCALE
            }
        '''
        cv2.IMREAD_COLOR: This is the default option, providing a 3-channel BGR image with an 8-bit value (0-255) for each channel.
        cv2.IMREAD_GRAYSCALE: This provides an 8-bit grayscale image.
        cv2.IMREAD_ANYCOLOR: This provides either an 8-bit-per-channel BGR image or an 8-bit grayscale image, depending on the metadata in the file.
        cv2.IMREAD_UNCHANGED: This reads all of the image data, including the alpha or transparency channel (if there is one) as a fourth channel.
        cv2.IMREAD_ANYDEPTH: This loads an image in grayscale at its original bit depth. For example, it provides a 16-bit-per-channel grayscale image if the file represents an image in this format.
        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR: This combination loads an image in BGR color at its original bit depth.
        cv2.IMREAD_REDUCED_GRAYSCALE_2: This loads an image in grayscale at half its original resolution. For example, if the file contains a 640 x 480 image, it is loaded as a 320 x 240 image.
        cv2.IMREAD_REDUCED_COLOR_2: This loads an image in 8-bit-per-channel BGR color at half its original resolution.
        cv2.IMREAD_REDUCED_GRAYSCALE_4: This loads an image in grayscale at one-quarter of its original resolution.
        cv2.IMREAD_REDUCED_COLOR_4: This loads an image in 8-bit-per-channel color at one-quarter of its original resolution.
        cv2.IMREAD_REDUCED_GRAYSCALE_8: This loads an image in grayscale at one-eighth of its original resolution.
        cv2.IMREAD_REDUCED_COLOR_8: This loads an image in 8-bit-per-channel color at one-eighth of its original resolution.
        '''
        if color_format not in map_.keys():
            raise Exception(f'Color format {color_format} not recognized.')
        if path[-5:] == '.tiff':
            success, images = cv2.imreadmulti(path, map_[color_format])
            if not success:
                raise Exception(f'Failed to read images located in {path}')
            return images
        else:
            image = cv2.imread(path, map_[color_format])
            if image is None:
                raise Exception(f'Failed to read image located in {path}')
            return image

    
    @staticmethod
    def edit_color_image(image, color_format):
        map_ = {
            'gray2bgr' : cv2.COLOR_GRAY2BGR,
            'bgr2rgb' : cv2.COLOR_BGR2RGB,
            'rgb2bgr' : cv2.COLOR_RGB2BGR,
            }
        if color_format not in map_.keys():
            raise Exception(f'Color format {color_format} not recognized.')
        image = cv2.cvtColor(image, map_[color_format])
        return image


    @staticmethod
    def edit_byte_image(image, width, height, byte):
        image.itemset((width, height), byte)
        '''(alternative) image[width, height] = byte'''
        return image


    @staticmethod
    def edit_color_image_full_roi(image, color):
        """Full region of interest"""
        if color=='blue':
            image[:, :, 0] = 255
        elif color=='green': 
            image[:, :, 1] = 255
        elif color=='red':
            image[:, :, 2] = 255
        return image
    
    
    @staticmethod
    def get_format(image):
        return {
            'shape': image.shape,
            'size': image.size,
            'dtype': image.dtype
            }