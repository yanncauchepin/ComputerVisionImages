import cv2


class FeatureDetection():
    
    
    '''Defining features
    What is a feature, exactly? Why is a particular area of an image classifiable as 
    a feature, while others are not? Broadly speaking, a feature is an area of interest 
    in the image that is unique or easily recognizable. Corners and regions with a high 
    density of textural detail are good features, while patterns that repeat themselves 
    a lot and low-density regions (such as a blue sky) are not. Edges are good features 
    as they tend to divide two regions with abrupt changes in the intensity (gray or color) 
    values of an image. A blob (a region of an image that greatly differs from its 
    surrounding areas) is also an interesting feature.
    Most feature detection algorithms revolve around the identification of corners, 
    edges, and blobs, with some also focusing on the concept of a ridge, which you can 
    conceptualize as the axis of symmetry of an elongated object. (Think, for example, 
    about identifying a road in an image.)
    '''
    
    
    def __init__(self):
        pass


    '''cv2.cornerHarris, is great for detecting corners and has a distinct advantage 
    because even if the image is rotated corners are still the corners. However, if 
    we scale an image to a smaller or larger size, some parts of the image may lose 
    or even gain a corner quality. You will notice how the corners are a lot more 
    condensed; however, even though we gained some corners, we lost others!'''
    @staticmethod
    def detecting_harris_corner(image, show=True):
        dst = cv2.cornerHarris(image, 2, 23, 0.04)
        '''The most important parameter here is the third one, which defines the aperture 
        or kernel size of the Sobel operator. The Sobel operator detects edges by 
        measuring horizontal and vertical differences between pixel values in a 
        neighborhood, and it does this using a kernel. The cv2.cornerHarris function 
        uses a Sobel operator whose aperture is defined by this parameter. In plain 
        English, the parameters define how sensitive corner detection is. It must be 
        between 3 and 31 and be an odd value. With a low (highly sensitive) value of 
        3, all those diagonal lines in the black squares of the chessboard will register 
        as corners when they touch the border of the square. For a higher (less sensitive) 
        value of 23, only the corners of each square will be detected as corners.
        cv2.cornerHarris returns an image in floating-point format. Each value in this 
        image represents a score for the corresponding pixel in the source image. A 
        moderate or high score indicates that the pixel is likely to be a corner. 
        Conversely, we can treat pixels with the lowest scores as non-corners.'''
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image[dst>0.01*dst.max()] = [0, 0, 255]
        '''Here, we select pixels with scores that are at least 1% of the highest 
        score, and we color these pixels red in the original image.'''
        if show:
            cv2.imshow('corners', image)
            cv2.waitKey(0)
        return image

