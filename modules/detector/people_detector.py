import cv2


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



class PeopleDetector():
    
    
    def __init__(self):
        pass


    '''Let's suppose we are training an SVM as a people detector. We have two classes, 
    person and non-person. As training samples, we provide vectors of HOG descriptors 
    of various windows that do or do not contain a person. These windows may come from 
    various images. The SVM learns by finding the optimal hyperplane that maximally 
    divides the multidimensional HOG descriptor space into people (on one side of the 
    hyperplane) and non-people (on the other side). Thereafter, when we give the trained 
    SVM a vector of HOG descriptors for any other window in any image, the SVM can 
    judge whether the window contains a person or not. The SVM can even give us a 
    confidence value that relates to the vector's distance from the optimal hyperplane.'''
    
    '''OpenCV comes with a class called cv2.HOGDescriptor, which is capable of performing 
    people detection. The interface has some similarities to the cv2.CascadeClassifier 
    class that we used in Chapter 5, Detecting and Recognizing Faces. However, unlike 
    cv2.CascadeClassifier, cv2.HOGDescriptor sometimes returns nested detection rectangles. 
    In other words, cv2.HOGDescriptor might tell us that it detected one person whose 
    bounding rectangle is located completely inside another person's bounding rectangle. 
    However, in a typical situation, nested detections are probably errors, so 
    cv2.HOGDescriptor is often used along with code to filter out any nested detections. 
    Let's begin our sample script by implementing a test to determine whether one 
    rectangle is nested inside another.
    '''
    @staticmethod
    def is_inside(i, o):
        '''
        For this purpose, we will write a function, is_inside(i, o), where i is the 
        possible inner rectangle and o is the possible outer rectangle.
        '''
        ix, iy, iw, ih = i
        ox, oy, ow, oh = o
        return ix > ox and ix + iw < ox + ow and \
            iy > oy and iy + ih < oy + oh
    
    @staticmethod
    def people_detector(image, show=True):
        '''
        Note that cv2.HOGDescriptor has a detectMultiScale method, which returns two 
        lists:
        - A list of bounding rectangles for detected objects (in this case, detected 
        people).
        - A list of weights or confidence scores for detected objects. A higher value 
        indicates greater confidence that the detection result is correct.
        
        detectMultiScale accepts several optional arguments, including the following:
        - winStride: This tuple defines the x and y distance that the sliding window 
        moves between successive detection attempts. HOG works well with overlapping 
        windows, so the stride may be small relative to the window size. A smaller 
        value produces more detections, at a higher computational cost. The default 
        stride has no overlap; it is the same as the window size, which is (64, 128) 
        for the default people detector.
        - scale: This scale factor is applied between successive levels of the image 
        pyramid. A smaller value produces more detections, at a higher computational 
        cost. The value must be greater than 1.0. The default is 1.5.
        - groupThreshold: This value determines how stringent our detection criteria 
        are. A smaller value is less stringent, resulting in more detections. The 
        default is 2.0.
    
        '''
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        '''Note that we specified the people detector with the setSVMDetector method.'''
        found_rects, found_weights = hog.detectMultiScale(
            image, winStride=(4, 4), scale=1.02, groupThreshold=1.9)
        found_rects_filtered = []
        found_weights_filtered = []
        for ri, r in enumerate(found_rects):
            for qi, q in enumerate(found_rects):
                if ri != qi and PeopleDetector.is_inside(r, q):
                    break
            else:
                found_rects_filtered.append(r)
                found_weights_filtered.append(found_weights[ri])
        for ri, r in enumerate(found_rects_filtered):
            x, y, w, h = r
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            text = '%.2f' % found_weights_filtered[ri]
            cv2.putText(image, text, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if show:
            cv2.imshow('People detector', image)
        return {
            "found_rects": found_rects,
            "founds_weight": found_weights,
            "found_rects_filtered": found_rects_filtered,
            "founds_weight_filtered": found_weights_filtered,
            "image": image
            }
    
    '''Sometimes, in the context of computer vision, BoW is called bag of visual words 
    (BoVW). BoW is the technique by which we assign a weight or count to each word in 
    a series of documents; we then represent these documents with vectors of these counts. 
    For example, a document can be classified as spam or not spam based on such a 
    representation. Indeed, spam filtering is one of the many real-world applications 
    of BoW.
    
    We are by now familiar with the concepts of features and descriptors. We have used 
    algorithms such as SIFT and SURF to extract descriptors from an image's features 
    so that we can match these features in another image.
    We have also recently familiarized ourselves with another kind of descriptor, based 
    on a codebook or dictionary. We know about an SVM, a model that can accept labeled 
    descriptor vectors as training data, can find an optimal division of the descriptor 
    space into the given classes, and can predict the classes of new data.
    Armed with this knowledge, we can take the following approach to build a classifier:
    1. Take a sample dataset of images.
    2. For each image in the dataset, extract descriptors (with SIFT, SURF, ORB, or 
    a similar algorithm).
    3. Add each descriptor vector to the BoW trainer.
    4. Cluster the descriptors into k clusters whose centers (centroids) are our visual 
    words. This last point probably sounds a bit obscure, but we will explore it further 
    in the next section.
    At the end of this process, we have a dictionary of visual words ready to be used. 
    As you can imagine, a large dataset will help make our dictionary richer in visual 
    words. Up to a point, the more words, the better!
    
    Given a test image, we can extract descriptors and quantize them (or reduce their 
    dimensionality) by calculating a histogram of their distances to the centroids. 
    Based on this, we can attempt to recognize visual words, and locate them in the image.
    
    k-means clustering is a method of quantization whereby we analyze a large number 
    of vectors in order to find a small number of clusters. Given a dataset, k represents 
    the number of clusters into which the dataset is going to be divided. The term means 
    refers to the mathematical concept of the mean or the average; when visually 
    represented, the mean of a cluster is its centroid or the geometric center of points 
    in the cluster. OpenCV provides a class called cv2.BOWKMeansTrainer, which we will 
    use to help train our classifier.
    '''

'''We are going to train a car detector, so our dataset must contain positive 
samples that represent cars, as well as negative samples that represent other 
(non-car) things that the detector is likely to encounter while looking for cars. 
For example, if the detector is intended to search for cars on a street, then a 
picture of a curb, a crosswalk, a pedestrian, or a bicycle might be a more 
representative negative sample than a picture of the rings of Saturn. Besides 
representing the expected subject matter, ideally, the training samples should 
represent the way our particular camera and algorithm will see the subject matter. 
We intend to use a sliding window of fixed size, so it is important that our 
training samples conform to a fixed size, and that the positive samples are 
tightly cropped in order to frame a car without much background.
'''