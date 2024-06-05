import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


class KeypointsDescriptors():
    
    
    '''Anatomy of a keypoint
    Each keypoint is an instance of the cv2.KeyPoint class, which has the following 
    properties:
    - The pt (point) property contains the x and y coordinates of the keypoint in 
    the image.
    - The size property indicates the diameter of the feature.
    - The angle property indicates the orientation of the feature, as shown by the 
    radial lines in the preceding processed image.
    - The response property indicates the strength of the keypoint. Some features 
    are classified by SIFT as stronger than others, and response is the property you 
    would check to evaluate the strength of a feature.
    - The octave property indicates the layer in the image pyramid where the feature 
    was found. Let's briefly review the concept of an image pyramid, which we discussed 
    previously in Chapter 5, Detecting and Recognizing Faces, in the Conceptualizing 
    Haar cascades section. The SIFT algorithm operates in a similar fashion to face 
    detection algorithms in that it processes the same image iteratively but alters 
    the input at each iteration. In particular, the scale of the image is a parameter 
    that changes at each iteration (octave) of the algorithm. Thus, the octave property 
    is related to the image scale at which the keypoint was detected.
    - Finally, the class_id property can be used to assign a custom identifier to a 
    keypoint or a group of keypoints.
    '''
    
    
    def __init__(self):
        pass


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
    @staticmethod
    def detecting_sift_keypoints_descriptors(image, show=True):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(image, keypoints, image, (51, 163, 236),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if show:
            cv2.imshow('sift_keypoints', image)
        return image
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


    '''SURF is several times faster than SIFT, and it is partially inspired by it.
    cv2.SURF is an OpenCV class that performs keypoint detection with the Fast Hessian 
    algorithm and descriptor extraction with SURF, much like the cv2.SIFT class performs 
    keypoint detection with DoG and descriptor extraction with SIFT.
    '''
    @staticmethod
    def detecting_surf_keypoints_descriptors(image, show=True):
        surf = cv2.xfeatures2d.SURF_create(8000)
        '''The parameter to cv2.xfeatures2d.SURF_create is a threshold for the Fast 
        Hessian algorithm. By increasing the threshold, we can reduce the number of 
        features that will be retained.'''
        keypoints, descriptor = surf.detectAndCompute(image, None)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawKeypoints(image, keypoints, image, (51, 163, 236),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if show:
            cv2.imshow('surf_keypoints', image)
        return image


    '''The Features from Accelerated Segment Test (FAST) algorithm works by analyzing 
    circular neighborhoods of 16 pixels. It marks each pixel in a neighborhood as 
    brighter or darker than a particular threshold, which is defined relative to the 
    center of the circle. A neighborhood is deemed to be a corner if it contains a 
    number of contiguous pixels marked as brighter or darker.
    FAST also uses a high-speed test, which can sometimes determine that a neighborhood 
    is not a corner by checking just 2 or 4 pixels instead of 16.
    Here, we can see a 16-pixel neighborhood at two different magnifications. The pixels 
    at positions 1, 5, 9, and 13 correspond to the four cardinal points at the edge 
    of the circular neighborhood. If the neighborhood is a corner, we expect that 
    out of these four pixels, exactly three or exactly one will be brighter than the 
    threshold. (Another way of saying this is that exactly one or exactly three of 
    them will be darker than the threshold.) If exactly two of them are brighter than 
    the threshold, then we have an edge, not a corner. If exactly four or exactly zero 
    of them are brighter than the threshold, then we have a relatively uniform 
    neighborhood that is neither a corner nor an edge.
    FAST is a clever algorithm, but it's not devoid of weaknesses, and to compensate 
    for these weaknesses, developers analyzing images can implement a machine learning 
    approach in order to feed a set of images (relevant to a given application) to the 
    algorithm so that parameters such as the threshold are optimized. Whether the 
    developer specifies parameters directly or provides a training set for a machine 
    learning approach, FAST is an algorithm that is sensitive to the developer's input, 
    perhaps more so than SIFT.'''
    
    '''Binary Robust Independent Elementary Features (BRIEF), on the other hand, is 
    not a feature detection algorithm, but a descriptor. Let's delve deeper into the 
    concept of what a descriptor is, and then look at BRIEF.
    When we previously analyzed images with SIFT and SURF, the heart of the entire 
    process was the call to the detectAndCompute function. This function performs two 
    different steps – detection and computation – and they return two different results, 
    coupled in a tuple.
    The result of detection is a set of keypoints; the result of the computation is a 
    set of descriptors for those keypoints. This means that OpenCV's cv2.SIFT and 
    cv2.xfeatures2d.SURF classes implement algorithms for both detection and description. 
    Remember, though, that the original SIFT and SURF are not feature detection algorithms. 
    OpenCV's cv2.SIFT implements DoG feature detection plus SIFT description, while 
    OpenCV's cv2.xfeatures2d.SURF implements Fast Hessian feature detection plus SURF 
    description.
    Keypoint descriptors are a representation of the image that serves as the gateway 
    to feature matching because you can compare the keypoint descriptors of two images 
    and find commonalities.
    BRIEF is one of the fastest descriptors currently available. The theory behind 
    BRIEF is quite complicated, but suffice it to say that BRIEF adopts a series of 
    optimizations that make it a very good choice for feature matching.
    '''
    
    '''ORB mixes the techniques used in the FAST keypoint detector and the BRIEF 
    keypoint descriptor.
    - The addition of a fast and accurate orientation component to FAST.
    - The efficient computation of oriented BRIEF features.
    - Analysis of variance and correlation of oriented BRIEF features.
    - A learning method to decorrelate BRIEF features under rotational invariance, 
    leading to better performance in nearest-neighbor applications.
    The main points are quite clear: ORB aims to optimize and speed up operations, 
    including the very important step of utilizing BRIEF in a rotation-aware fashion 
    so that matching is improved, even in situations where a training image has a 
    very different rotation to the query image.
    '''