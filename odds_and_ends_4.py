import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


'''HOG is a feature descriptor, so it belongs to the same family of algorithms
SIFT, SURF, Oriented FAST and ORB. Like other feature descriptors, HOG is capable 
of delivering the type of information that is vital for feature matching, as well 
as for object detection and recognition. Most commonly, HOG is used for object 
detection. HOG's internal mechanism is really clever; an image is divided into 
cells and a set of gradients is calculated for each cell. Each gradient describes 
the change in pixel intensities in a given direction. Together, these gradients 
form a histogram representation of the cell. We encountered a similar approach 
when we studied face recognition with the local binary pattern histogram (LBPH).

For each HOG cell, the histogram contains a number of bins equal to the number of 
gradients or, in other words, the number of axis directions that HOG considers. 
After calculating all the cells' histograms, HOG processes groups of histograms 
to produce higher-level descriptors. Specifically, the cells are grouped into larger 
regions, called blocks. These blocks can be made of any number of cells, but Dalal 
and Triggs found that 2x2 cell blocks yielded the best results when performing 
people detection. A block-wide vector is created so that it can be normalized, 
compensating for local variations in illumination and shadowing. (A single cell 
is too small a region to detect such variations.) This normalization improves a 
HOG-based detector's robustness, with respect to variations in lighting conditions.
Like other detectors, a HOG-based detector needs to cope with variations in objects' 
location and scale. The need to search in various locations is addressed by moving 
a fixed-size sliding window across an image. The need to search at various scales 
is addressed by scaling the image to various sizes, forming a so-called image 
pyramid.

Suppose we are using a sliding window to perform people detection on an image. 
We slide our window in small steps, just a few pixels at a time, so we expect 
that it will frame any given person multiple times. Assuming that overlapping 
detections are indeed one person, we do not want to report multiple locations but, 
rather, only one location that we believe to be correct. In other words, even if 
a detection at a given location has a good confidence score, we might reject it 
if an overlapping detection has a better confidence score; thus, from a set of 
overlapping detections, we would choose the one with the best confidence score.
This is where NMS comes into play. Given a set of overlapping regions, we can 
suppress (or reject) all the regions for which our classifier did not produce a 
maximal score.'''


'''The concept of NMS might sound simple. From a set of overlapping solutions, 
just pick the best one! However, the implementation is more complex than you might 
initially think. Remember the image pyramid? Overlapping detections can occur at 
different scales. We must gather up all our positive detections, and convert their 
bounds back to a common scale before we check for overlap. A typical implementation 
of NMS takes the following approach:
- Construct an image pyramid.
- Scan each level of the pyramid with the sliding window approach, for object 
detection. For each window that yields a positive detection (beyond a certain 
arbitrary confidence threshold), convert the window back to the original image's 
scale. Add the window and its confidence score to a list of positive detections.
- Sort the list of positive detections by order of descending confidence score 
so that the best detections come first in the list.
- For each window, W, in the list of positive detections, remove all subsequent 
windows that significantly overlap with W. We are left with a list of positive 
detections that satisfy the criterion of NMS.
Besides NMS, another way to filter the positive detections is to eliminate any 
subwindows. When we speak of a subwindow (or subregion), we mean a window (or 
region in an image) that is entirely contained inside another window (or region). 
To check for subwindows, we simply need to compare the corner coordinates of various 
window rectangles. We will take this simple approach in our first practical example, 
in the Detecting people with HOG descriptors section. Optionally, NMS and suppression 
of subwindows can be combined.'''

'''Let's suppose we are training an SVM as a people detector. We have two classes, 
person and non-person. As training samples, we provide vectors of HOG descriptors 
of various windows that do or do not contain a person. These windows may come from 
various images. The SVM learns by finding the optimal hyperplane that maximally 
divides the multidimensional HOG descriptor space into people (on one side of the 
hyperplane) and non-people (on the other side). Thereafter, when we give the trained 
SVM a vector of HOG descriptors for any other window in any image, the SVM can 
judge whether the window contains a person or not. The SVM can even give us a 
confidence value that relates to the vector's distance from the optimal hyperplane.''

if __name__=='__main__':
    pass