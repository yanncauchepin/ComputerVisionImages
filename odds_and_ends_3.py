import os
import cv2
import numpy as np

"""FEATURES DETECTION, MATCHING AND IMAGE DESCRIPTOR"""

'''
A number of algorithms can be used to detect and describe features, and we will 
explore several of them in this section. The most commonly used feature detection 
and descriptor extraction algorithms in OpenCV are as follows:

- Harris: This algorithm is useful for detecting corners.
- SIFT: This algorithm is useful for detecting blobs.
- SURF: This algorithm is useful for detecting blobs.
- FAST: This algorithm is useful for detecting corners.
- BRIEF: This algorithm is useful for detecting blobs.
- ORB: This algorithm stands for Oriented FAST and Rotated BRIEF. It is useful for detecting a combination of corners and blobs.

Matching features can be performed with the following methods:

- Brute-force matching
- FLANN-based matching

Spatial verification can then be performed with homography.
'''

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


'''cv2.cornerHarris, is great for detecting corners and has a distinct advantage 
because even if the image is rotated corners are still the corners. However, if 
we scale an image to a smaller or larger size, some parts of the image may lose 
or even gain a corner quality. You will notice how the corners are a lot more 
condensed; however, even though we gained some corners, we lost others!'''
def detecting_harris_corner(image):
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
    cv2.imshow('corners', image)
    

'''Scale-Invariant Feature Transform (SIFT). While the name may sound a bit mysterious, 
now that we know what problem we are trying to solve, it actually makes sense. We 
need a function (a transform) that will detect features (a feature transform) and 
will not output different results depending on the scale of the image 
(a scale-invariant feature transform). Note that SIFT does not detect keypoints. 
Keypoint detection is done with the Difference of Gaussians (DoG), while SIFT 
describes the region surrounding the keypoints by means of a feature vector.
DoG is the result of applying different Gaussian filters to the same image. 
Previously, we applied this type of technique for edge detection, and the idea 
is the same here. The final result of a DoG operation contains areas of interest 
(keypoints), which are then going to be described through SIFT.'''
def detecting_sift_keypoints_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(image, keypoints, image, (51, 163, 236),
                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('sift_keypoints', image)
    '''Behind the scenes, these simple lines of code carry out an elaborate process; 
    we create a cv2.SIFT object, which uses DoG to detect keypoints and then computes 
    a feature vector for the surrounding region of each keypoint. As the name of 
    the detectAndCompute method clearly suggests, two main operations are performed: 
    feature detection and the computation of descriptors. The return value of the 
    operation is a tuple containing a list of keypoints and another list of the 
    keypoints' descriptors.
    Finally, we process this image by drawing the keypoints on it with the 
    cv2.drawKeypoints function and then displaying it with the usual cv2.imshow 
    function. As one of its arguments, the cv2.drawKeypoints function accepts a 
    flag that specifies the type of visualization we want. Here, we specify 
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINT in order to draw a visualization of 
    the scale and orientation of each keypoint.'''


if __name__ == '__main__':
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/chessboard.png'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detecting_harris_corner(image_gray)
    image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/Thoune3.jpg'
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detecting_sift_keypoints_descriptors(image_gray)
    