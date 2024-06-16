import cv2
import numpy as np
import os

'''The preceding function takes an image and generates a series of resized versions 
of it. The series is bounded by a maximum and minimum image size.
You will have noticed that the resized image is not returned with the return keyword 
but with the yield keyword. This is because this function is a so-called generator. 
It produces a series of images that we can easily use in a loop. If you are not 
familiar with generators, take a look at the official Python Wiki at 
https://wiki.python.org/moin/Generators.'''
def pyramid(img, scale_factor=1.05, min_size=(100, 40), max_size=(600, 240)):
    # Get the dimensions of the input image
    h, w = img.shape

    # Extract minimum and maximum size constraints
    min_w, min_h = min_size
    max_w, max_h = max_size

    # Loop while the image dimensions are within the specified range
    while w >= min_w and h >= min_h:
        # Check if the current size is within the maximum size limit
        if w <= max_w and h <= max_h:
            # If within limits, yield the current resized image
            yield img

        # Reduce the width and height by the scale factor
        w /= scale_factor
        h /= scale_factor

        # Resize the image using OpenCV
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)


'''Again, this is a generator. Although it is a bit deep-nested, the mechanism is 
very simple: given an image, return the upper-left coordinates and the sub-image 
representing the next window. Successive windows are shifted by an arbitrarily 
sized step from left to right until we reach the end of a row, and from the top 
to bottom until we reach the end of the image.'''
def sliding_window(img, step=20, window_size=(100, 40)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w, step):
        for x in range(0, img_h, step):
            roi = img[y:y+window_h, x:x+window_w]
            roi_h, roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield (x, y, roi)


'''As its first argument, the function takes a NumPy array containing rectangle 
coordinates and scores. If we have N rectangles, the shape of this array is Nx5. 
For a given rectangle at index i, the values in the array have the following meanings:
    boxes[i][0] is the leftmost x coordinate.
    boxes[i][1] is the topmost y coordinate.
    boxes[i][2] is the rightmost x coordinate.
    boxes[i][3] is the bottommost y coordinate.
    boxes[i][4] is the score, where a higher score represents greater confidence 
    that the rectangle is a correct detection result.

As its second argument, the function takes a threshold that represents the maximum 
proportion of overlap between rectangles. If two rectangles have a greater proportion 
of overlap than this, the one with the lower score will be filtered out. Ultimately, 
the function will return an array of the remaining rectangles.
'''
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes 
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score/probability of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick]

def detector_car():

    root_path = '/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionImages/classifier_car'
    
    BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
    SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 11
    BOW_NUM_CLUSTERS = 40
    
    '''Note that our classifier will make use of two training stages: one stage for 
    the BoW vocabulary, which will use a number of images as samples, and another stage 
    for the SVM, which will use a number of BoW descriptor vectors as samples. Arbitrarily, 
    we have defined a different number of training samples for each stage. At each stage, 
    we could have also defined a different number of training samples for the two classes 
    (car and not car), but instead, we will use the same number. We have also defined 
    the number of BoW clusters. Note that the number of BoW clusters may be larger than 
    the number of BoW training images, since each image has many descriptors.
    '''
    
    '''We will use cv2.SIFT to extract descriptors and cv2.FlannBasedMatcher to match 
    these descriptors. Note that we have initialized SIFT and the Fast Library for 
    Appropriate Nearest Neighbors (FLANN) Let's initialize these algorithms with the 
    following code:'''
    
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    '''OpenCV provides a class called cv2.BOWKMeansTrainer to train a BoW vocabulary, 
    and a class called cv2.BOWImgDescriptorExtractor to convert some kind of lower-level 
    descriptors – in our example, SIFT descriptors – into BoW descriptors. Let's 
    initialize these objects with the following code:'''
    
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(BOW_NUM_CLUSTERS)
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)
    
    '''When initializing cv2.BOWKMeansTrainer, we must specify the number of clusters 
    – in our example, 40, as defined earlier. When initializing cv2.BOWImgDescriptorExtractor, 
    we must specify a descriptor extractor and a descriptor matcher – in our example, 
    he cv2.SIFT and cv2.FlannBasedMatcher objects that we created earlier.'''
    
    '''To train the BoW vocabulary, we will provide samples of SIFT descriptors for 
    various car and not car images. We will load the images from the CarData/TrainImages 
    subfolder, which contains positive (car) images with names such as pos-x.pgm, and 
    negative (not car) images with names such as neg-x.pgm, where x is a number starting 
    at 1. Let's write the following utility function to return a pair of paths to the 
    ith positive and negative training images, where i is a number starting at 0:'''
    def get_pos_and_neg_paths(i):
        pos_path = root_path + '/TrainImages/pos-%d.pgm' % (i+1)
        neg_path = root_path + '/TrainImages/neg-%d.pgm' % (i+1)
        return pos_path, neg_path
    
    '''Later in this section, we will call the preceding function in a loop, with a 
    varying value of i, when we need to acquire a number of training samples.'''
    
    '''For each path to a training sample, we will need to load the image, extract 
    SIFT descriptors, and add the descriptors to the BoW vocabulary trainer. Let's 
    write another utility function to do precisely this, as follows:'''
    def add_sample(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            bow_kmeans_trainer.add(descriptors)
    
    '''At this stage, we have everything we need to start training the BoW vocabulary. 
    Let's read a number of images for each class (car as the positive class and not 
    car as the negative class) and add them to the training set, as follows:'''
    for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
        pos_path, neg_path = get_pos_and_neg_paths(i)
        add_sample(pos_path)
        add_sample(neg_path)
        
    '''Now that we have assembled the training set, we will call the vocabulary trainer's 
    cluster method, which performs the k-means classification and returns the vocabulary. 
    We will assign this vocabulary to the BoW descriptor extractor, as follows:'''
    voc = bow_kmeans_trainer.cluster()
    bow_extractor.setVocabulary(voc)
    
    '''Remember that earlier, we initialized the BoW descriptor extractor with a SIFT 
    descriptor extractor and FLANN matcher. Now, we have also given the BoW descriptor 
    extractor a vocabulary that we trained with samples of SIFT descriptors. At this 
    stage, our BoW descriptor extractor has everything it needs in order to extract 
    BoW descriptors from Difference of Gaussian (DoG) features. Remember that cv2.SIFT 
    detects DoG features and extracts SIFT descriptors.
    '''
    
    '''Next, we will declare another utility function that takes an image and returns 
    the descriptor vector, as computed by the BoW descriptor extractor. This involves 
    extracting the image's DoG features, and computing the BoW descriptor vector based 
    on the DoG features, as follows:'''
    def extract_bow_descriptors(image):
        features = sift.detect(image)
        return bow_extractor.compute(image, features)
    
    '''We are ready to assemble another kind of training set, containing samples of 
    BoW descriptors. Let's create two arrays to accommodate the training data and 
    labels, and populate them with the descriptors generated by our BoW descriptor 
    extractor. We will label each descriptor vector with 1 for a positive sample and 
    -1 for a negative sample, as shown in the following code block:'''
    training_data = []
    training_labels = []
    for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
        pos_path, neg_path = get_pos_and_neg_paths(i)
        pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
        pos_descriptors = extract_bow_descriptors(pos_img)
        if pos_descriptors is not None:
            training_data.extend(pos_descriptors)
            training_labels.append(1)
        neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
        neg_descriptors = extract_bow_descriptors(neg_img)
        if neg_descriptors is not None:
            training_data.extend(neg_descriptors)
            training_labels.append(-1)
    
    '''Should you wish to train a classifier to distinguish between multiple positive 
    classes, you can simply add other descriptors with other labels. For example, we 
    could train a classifier that uses the label 1 for car, 2 for person, and -1 for 
    background. There is no requirement to have a negative or background class but, 
    if you do not, your classifier will assume that everything belongs to one of the 
    positive classes.
    '''
    '''OpenCV provides a class called cv2.ml_SVM, representing an SVM. Let's create 
    an SVM, and train it with the data and labels that we previously assembled, as 
    follows:
    '''
    svm = cv2.ml.SVM_create()
    svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
              np.array(training_labels))
    
    '''Note that we must convert the training data and labels from lists to NumPy 
    arrays before we pass them to the train method of cv2.ml_SVM.'''
    
    '''Finally, we are ready to test the SVM by classifying some images that were not 
    part of the training set. We will iterate over a list of paths to test images. 
    For each path, we will load the image, extract BoW descriptors, and get the SVM's 
    prediction or classification result, which will be either 1.0 (car) or -1.0 (not car), 
    based on the training labels we used earlier. We will draw text on the image to 
    show the classification result, and we will show the image in a window. After 
    showing all the images, we will wait for the user to hit any key, and then the 
    script will end. All of this is achieved in the following block of code:'''
    for test_img_path in [root_path + '/TestImages/test-0.pgm',
                          root_path + '/TestImages/test-1.pgm',
                          '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/car.jpg',
                          '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/cars_small.jpg']:
        img = cv2.imread(test_img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        descriptors = extract_bow_descriptors(gray_img)
        prediction = svm.predict(descriptors)
        if prediction[1][0][0] == 1.0:
            text = 'car'
            color = (0, 255, 0)
        else:
            text = 'not car'
            color = (0, 0, 255)
        cv2.putText(img, text, (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)
        
    '''
    Incorrect classification solely depends upon the amount and nature of positive 
    and negative images used for training and the robustness of the learning algorithm. 
    For example, if the car position in every image of our training dataset is in the 
    center of the image, the test image with a car position at the corner of the 
    image may classify incorrectly.
    Experiment with adjusting the number of training samples and other parameters, 
    and try testing the classifier on more images, to see what results you can get.
    Let's take stock of what we have done so far. We have used a mixture of SIFT, 
    BoW, and SVMs to train a classifier to distinguish between two classes: car 
    and not car. We have applied this classifier to whole images. The next logical 
    step is to apply a sliding window technique so that we can narrow down our 
    classification results to specific regions of an image.
    '''

def detector_car_sliding_window()    :
    '''
    1. Take a region of the image, classify it, and then move this window to the 
    right by a predefined step size. When we reach the rightmost end of the image, 
    reset the x coordinate to 0, move down a step, and repeat the entire process.
    2. At each step, perform a classification with the SVM that was trained with BoW.
    3. Keep track of all the windows that are positive detections, according to the SVM.
    4. After classifying every window in the entire image, scale the image down, 
    and repeat the entire process of using a sliding window. Thus, we are using 
    an image pyramid. Continue rescaling and classifying until we get to a minimum 
    size.
    
    When we reach the end of this process, we have collected important information 
    about the content of the image. However, there is a problem: in all likelihood, 
    we have found a number of overlapping blocks that each yield a positive detection 
    with high confidence. That is to say, the image may contain one object that gets 
    detected multiple times. If we reported these multiple detections, our report 
    would be quite misleading, so we will filter our results using NMS.
    
    For NMS, we will rely on Malisiewicz and Rosebrock's implementation.
    '''
    root_path = '/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionImages/classifier_car'
    
    BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
    SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 110

    BOW_NUM_CLUSTERS = 12
    SVM_SCORE_THRESHOLD = 2.2
    NMS_OVERLAP_THRESHOLD = 0.4

    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    bow_kmeans_trainer = cv2.BOWKMeansTrainer(BOW_NUM_CLUSTERS)
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

    def get_pos_and_neg_paths(i):
        pos_path = root_path + '/TrainImages/pos-%d.pgm' % (i+1)
        neg_path = root_path + '/TrainImages/neg-%d.pgm' % (i+1)
        return pos_path, neg_path

    def add_sample(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            bow_kmeans_trainer.add(descriptors)

    for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
        pos_path, neg_path = get_pos_and_neg_paths(i)
        add_sample(pos_path)
        add_sample(neg_path)

    voc = bow_kmeans_trainer.cluster()
    bow_extractor.setVocabulary(voc)

    def extract_bow_descriptors(img):
        features = sift.detect(img)
        return bow_extractor.compute(img, features)

    training_data = []
    training_labels = []
    for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
        pos_path, neg_path = get_pos_and_neg_paths(i)
        pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
        pos_descriptors = extract_bow_descriptors(pos_img)
        if pos_descriptors is not None:
            training_data.extend(pos_descriptors)
            training_labels.append(1)
        neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
        neg_descriptors = extract_bow_descriptors(neg_img)
        if neg_descriptors is not None:
            training_data.extend(neg_descriptors)
            training_labels.append(-1)
    
    """With the preceding changes to the SVM, we are specifying the classifier's 
    level of strictness or severity. As the value of the C parameter increases, 
    the risk of false positives decreases but the risk of false negatives increases. 
    In our application, a false positive would be a window detected as a car when 
    it is really not a car, and a false negative would be a car detected as a window 
    when it really is a car.

    After the code that trains the SVM, we want to add two more helper functions. 
    One of them will generate levels of the image pyramid, and the other will generate 
    regions of interest, based on the sliding window technique. Besides adding these 
    helper functions, we also need to handle the test images differently in order 
    to make use of the sliding window and NMS. The following steps cover the changes:"""
    
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(50)

    svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
              np.array(training_labels))

    def pyramid(img, scale_factor=1.05, min_size=(100, 40),
                max_size=(600, 240)):
        h, w = img.shape
        min_w, min_h = min_size
        max_w, max_h = max_size
        while w >= min_w and h >= min_h:
            if w <= max_w and h <= max_h:
                yield img
            w /= scale_factor
            h /= scale_factor
            img = cv2.resize(img, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA)

    def sliding_window(img, step=20, window_size=(100, 40)):
        img_h, img_w = img.shape
        window_w, window_h = window_size
        for y in range(0, img_w, step):
            for x in range(0, img_h, step):
                roi = img[y:y+window_h, x:x+window_w]
                roi_h, roi_w = roi.shape
                if roi_w == window_w and roi_h == window_h:
                    yield (x, y, roi)

    for test_img_path in [root_path + '/TestImages/test-0.pgm',
                          root_path + '/TestImages/test-1.pgm',
                          '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/car.jpg',
                          '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/cars_small.jpg']:
        img = cv2.imread(test_img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''For each test image, we iterate over the pyramid levels, and for each pyramid 
        level, we iterate over the sliding window positions. For each window or region 
        of interest (ROI), we extract BoW descriptors and classify them using the SVM. 
        If the classification produces a positive result that passes a certain confidence 
        threshold, we add the rectangle's corner coordinates and confidence score to 
        a list of positive detections. Continuing from the previous code block, we 
        proceed to handle a given test image with the following code:'''
        pos_rects = []
        for resized in pyramid(gray_img):
            for x, y, roi in sliding_window(resized):
                descriptors = extract_bow_descriptors(roi)
                if descriptors is None:
                    continue
                prediction = svm.predict(descriptors)
                if prediction[1][0][0] == 1.0:
                    raw_prediction = svm.predict(
                        descriptors, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                    score = -raw_prediction[1][0][0]
                    if score > SVM_SCORE_THRESHOLD:
                        h, w = roi.shape
                        scale = gray_img.shape[0] / float(resized.shape[0])
                        pos_rects.append([int(x * scale),
                                          int(y * scale),
                                          int((x+w) * scale),
                                          int((y+h) * scale),
                                          score])
        '''To obtain a confidence score for the SVM's prediction, we must run the predict 
        method with an optional flag, cv2.ml.STAT_MODEL_RAW_OUTPUT. Then, instead of 
        returning a label, the method returns a score as part of its output. This 
        score may be negative, and a low value represents a high level of confidence. 
        To make the score more intuitive – and to match the NMS function's assumption 
        that a higher score is better – we negate the score so that a high value 
        represents a high level of confidence.
        Since we are working with multiple pyramid levels, the window coordinates do 
        not have a common scale. We have converted them back to a common scale – the 
        original image's scale – before adding them to our list of positive detections.'''
        pos_rects = non_max_suppression_fast(
            np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
        '''Note that we have converted our list of rectangle coordinates and scores 
        to a NumPy array, which is the format expected by this function.
        At this stage, we have an array of detected car rectangles and their scores, 
        and we have ensured that these are the best non-overlapping detections we 
        can select (within the parameters of our model).'''
        for x0, y0, x1, y1, score in pos_rects:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 255, 255), 2)
            text = '%.2f' % score
            cv2.putText(img, text, (int(x0), int(y0) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(test_img_path, img)
        
    '''Remember that in this sample script, our training sets are small. Larger 
    training sets, with more diverse backgrounds, could improve the results. Also, 
    remember that the image pyramid and sliding window are producing a large number 
    of ROIs. When we consider this, we should realize that aa low false positive 
    rate is actually a significant accomplishment. Moreover, if we were performing 
    detection on frames of a video, we could mitigate the problem of false negatives 
    because we would have a chance to detect a car in multiple frames.'''
  

if __name__=='__main__':
    detector_car_sliding_window()