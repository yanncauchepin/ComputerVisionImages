import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

'''SURF is several times faster than SIFT, and it is partially inspired by it.
cv2.SURF is an OpenCV class that performs keypoint detection with the Fast Hessian 
algorithm and descriptor extraction with SURF, much like the cv2.SIFT class performs 
keypoint detection with DoG and descriptor extraction with SIFT.
'''
def detecting_surf_keypoints_descriptors(image):
    surf = cv2.xfeatures2d.SURF_create(8000)
    '''The parameter to cv2.xfeatures2d.SURF_create is a threshold for the Fast 
    Hessian algorithm. By increasing the threshold, we can reduce the number of 
    features that will be retained.'''
    keypoints, descriptor = surf.detectAndCompute(image, None)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(image, keypoints, image, (51, 163, 236),
                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('surf_keypoints', image)


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

'''Brute-force matching
A brute-force matcher is a descriptor matcher that compares two sets of keypoint 
descriptors and generates a result that is a list of matches. It is called brute-force 
because little optimization is involved in the algorithm. For each keypoint 
descriptor in the first set, the matcher makes comparisons to every keypoint 
descriptor in the second set. Each comparison produces a distance value and the 
best match can be chosen on the basis of least distance.
More generally, in computing, the term brute-force is associated with an approach 
that prioritizes the exhaustion of all possible combinations (for example, all the 
possible combinations of characters to crack a password of a known length). 
Conversely, an algorithm that prioritizes speed might skip some possibilities and 
try to take a shortcut to the solution that seems the most plausible.
OpenCV provides a cv2.BFMatcher class that supports several approaches to 
brute-force feature matching.
'''

'''matching a logo in two images'''
def orb_matching(image_1, image_2):
    orb = cv2.ORB_create()
    image_1_kp, image_1_descriptor = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_descriptor = orb.detectAndCompute(image_2, None)
    # Perform brute-force matching.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(image_1_descriptor, image_2_descriptor)
    # Sort the matches by distance.
    matches = sorted(matches, key=lambda x:x.distance)
    # Draw the best 25 matches.
    image_matches = cv2.drawMatches(
        image_1, image_1_kp, image_2, image_2_kp, matches[:25], image_2,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # Show the matches.
    plt.imshow(image_matches)
    plt.show()
        

'''Now, let's consider the implementation of a modified brute-force matching algorithm 
that adaptively chooses a distance threshold in the manner we have described. In 
the previous section's code sample, we used the match method of the cv2.BFMatcher 
class in order to get a list containing the single best (least-distance) match for 
each query keypoint. By doing so, we discarded information about the distance 
scores of all the worse possible matches – the kind of information we need for 
our adaptive approach. Fortunately, cv2.BFMatcher also provides a knnMatch method, 
which accepts an argument, k, that specifies the maximum number of best (least-distance) 
matches that we want to retain for each query keypoint. (In some cases, we may 
get fewer matches than the maximum.) KNN stands for k-nearest neighbors.
We will use the knnMatch method to request a list of the two best matches for each 
query keypoint. Based on our assumption that each query keypoint has, at most, one 
correct match, we are confident that the second-best match is wrong. We multiply 
the second-best match's distance score by a value less than 1 in order to obtain 
the threshold.
Then, we accept the best match as a good match only if its distant score is less 
than the threshold. This approach is known as the ratio test.
"The probability that a match is correct can be determined by taking the ratio 
of the distance from the closest neighbor to the distance of the second closest." 
'''


def orb_knn_matching(image_1, image_2):
    orb = cv2.ORB_create()
    image_1_kp, image_1_descriptor = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_descriptor = orb.detectAndCompute(image_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs_of_matches = bf.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
    '''knnMatch returns a list of lists; each inner list contains at least one 
    match and no more than k matches, sorted from best (least distance) to worst. 
    The following line of code sorts the outer list based on the distance score 
    of the best matches'''
    pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)
    '''Draw the top 25 best matches, along with any second-best matches that knnMatch 
    may have paired with them. We can't use the cv2.drawMatches function because 
    it only accepts a one-dimensional list of matches; instead, we must use 
    cv2.drawMatchesKnn.'''
    image_pairs_of_matches = cv2.drawMatchesKnn(
        image_1, image_1_kp, image_2, image_2_kp, pairs_of_matches[:25], image_2,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(image_pairs_of_matches)
    plt.show()
    '''We have not filtered out any bad matches – and, indeed, we have deliberately 
    included the second-best matches, which we believe to be bad.'''


def orb_knn_ratio_test_matching(image_1, image_2):
    '''Set the threshold at 0.8 times the distance score of the second-best match. 
    If knnMatch has failed to provide a second-best match, we reject the best match 
    anyway because we are unable to apply the test.'''
    orb = cv2.ORB_create()
    image_1_kp, image_1_descriptor = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_descriptor = orb.detectAndCompute(image_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs_of_matches = bf.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
    pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)
    # Apply the ratio test.
    matches = [x[0] for x in pairs_of_matches if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]
    image_matches = cv2.drawMatches(
        image_1, image_1_kp, image_2, image_2_kp, matches[:25], image_2,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(image_matches)
    plt.show()


'''FLANN stands for Fast Library for Approximate Nearest Neighbors. "FLANN is a 
library for performing fast approximate nearest neighbor searches in high dimensional 
spaces. It contains a collection of algorithms we found to work best for the nearest 
neighbor search and a system for automatically choosing the best algorithm and 
optimum parameters depending on the dataset. FLANN is written in C++ and contains 
bindings for the following languages: C, MATLAB,Python, and Ruby.". Although FLANN 
is also available as a standalone library, we will use it as part of OpenCV because 
OpenCV provides a handy wrapper for it.'''

def flann_matching(image_1, image_2):
    sift = cv2.SIFT_create()
    image_1_kp, image_1_descriptor = sift.detectAndCompute(image_1, None)
    image_2_kp, image_2_descriptor = sift.detectAndCompute(image_2, None)
    # Define FLANN-based matching parameters.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # Perform FLANN-based matching.
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
    '''Here, we can see that the FLANN matcher takes two parameters: an indexParams 
    object and a searchParams object. These parameters, passed in the form of 
    dictionaries in Python (and structs in C++), determine the behavior of the 
    index and search objects that are used internally by FLANN to compute the matches. 
    We have chosen parameters that offer a reasonable balance between accuracy and 
    processing speed. Specifically, we are using a kernel density tree (kd-tree) 
    indexing algorithm with five trees, which FLANN can process in parallel. 
    (The FLANN documentation recommends between one tree, which would offer no 
    parallelism, and 16 trees, which would offer a high degree of parallelism if 
    the system could exploit it.)
    We are performing 50 checks or traversals of each tree. A greater number of 
    checks can provide greater accuracy but at a greater computational cost.'''
    # Prepare an empty mask to draw good matches.
    mask_matches = [[0, 0] for i in range(len(matches))]
    # Populate the mask based on David G. Lowe's ratio test.
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            mask_matches[i]=[1, 0]
    '''After performing FLANN-based matching, we apply Lowe's ratio test with a 
    multiplier of 0.7. To demonstrate a different coding style, we will use the 
    result of the ratio test in a slightly different way compared to how we did 
    in the previous section's code sample. Previously, we assembled a new list 
    with just the good matches in it. This time, we will assemble a list called 
    mask_matches, in which each element is a sublist of length k (the same k 
    that we passed to knnMatch). If a match is good, we set the corresponding 
    element of the sublist to 1; otherwise, we set it to 0.
    For example, if we have mask_matches = [[0, 0], [1, 0]], this means that we 
    have two matched keypoints; for the first keypoint, the best and second-best 
    matches are both bad, while for the second keypoint, the best match is good 
    but the second-best match is bad. Remember, we assume that all the second-best 
    matches are bad.'''
    '''It is time to draw and show the good matches. We can pass our mask_matches 
    list to cv2.drawMatchesKnn as an optional argument, as highlighted.'''
    # Draw the matches that passed the ratio test.
    image_matches = cv2.drawMatchesKnn(
        image_1, image_1_kp, image_2, image_2_kp, matches, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        matchesMask=mask_matches, flags=0)
    '''cv2.drawMatchesKnn only draws the matches that we marked as good (with a 
    value of 1) in our mask. Let's unveil the result.'''
    # Show the matches.
    plt.imshow(image_matches)
    plt.show()


'''Homography: "A relation between two figures, such that to any point of the one 
corresponds one and but one point in the other, and vice versa. Thus, a tangent 
line rolling on a circle cuts two fixed tangents of the circle in two sets of 
points that are homographic.". A bit clearer: homography is a condition in which 
two figures find each other when one is a perspective distortion of the other.'''

if __name__ == '__main__':
    
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/chessboard.png'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detecting_harris_corner(image_gray)
    
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/Thoune3.jpg'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detecting_surf_keypoints_descriptors(image_gray)
    
    # feature_image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/nasa_logo.png'
    # feature_image = cv2.imread(feature_image_path, cv2.IMREAD_GRAYSCALE)
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/kennedy_space_center.jpg'
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image_knn_ratio_test_matching(feature_image, image)
    
    feature_image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/gauguin_entre_les_lys.jpg'
    feature_image = cv2.imread(feature_image_path, cv2.IMREAD_GRAYSCALE)
    image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/gauguin_paintings.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    flann_matching(feature_image, image)
    