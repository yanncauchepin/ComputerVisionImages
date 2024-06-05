import cv2
import numpy as np
from random import randint, uniform 
import gzip 
import pickle 
import mediapipe as mp
import landmark_utils as u
import copy
import csv
import itertools
from sklearn.model_selection import train_test_split
import tensorflow as tf

def ann_sample():
    
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([9, 15, 9], np.uint8)) 
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0) 
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1) 
    ann.setTermCriteria( 
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0)) 
    
    training_samples = np.array( 
        [[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], np.float32) 
    layout = cv2.ml.ROW_SAMPLE 
    training_responses = np.array( 
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], np.float32) 
    data = cv2.ml.TrainData_create( 
        training_samples, layout, training_responses) 
    ann.train(data)
    
    test_samples = np.array( 
        [[1.4, 1.5, 1.2, 2.0, 2.5, 2.8, 3.0, 3.1, 3.8]], np.float32) 
    prediction = ann.predict(test_samples) 
    print(prediction) 


def record(sample, classification): 
    return (np.array([sample], np.float32), 
          np.array([classification], np.float32)) 

def animal_classification_ann():

    animals_net = cv2.ml.ANN_MLP_create() 
    animals_net.setLayerSizes(np.array([3, 50, 4])) 
    animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0) 
    animals_net.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1) 
    animals_net.setTermCriteria( 
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0)) 
    
    """Input arrays 
    weight, length, teeth 
    """ 
    """Output arrays 
    dog, condor, dolphin, dragon 
    """ 
    def dog_sample(): 
        return [uniform(10.0, 20.0), uniform(1.0, 1.5), 
            randint(38, 42)] 
    def dog_class(): 
        return [1, 0, 0, 0] 
    def condor_sample(): 
        return [uniform(3.0, 10.0), randint(3.0, 5.0), 0] 
    def condor_class(): 
        return [0, 1, 0, 0] 
    def dolphin_sample(): 
        return [uniform(30.0, 190.0), uniform(5.0, 15.0),  
            randint(80, 100)] 
    def dolphin_class(): 
        return [0, 0, 1, 0] 
    def dragon_sample(): 
        return [uniform(1200.0, 1800.0), uniform(30.0, 40.0),  
            randint(160, 180)] 
    def dragon_class(): 
        return [0, 0, 0, 1] 
    
    RECORDS = 20000 
    records = [] 
    for x in range(0, RECORDS): 
        records.append(record(dog_sample(), dog_class())) 
        records.append(record(condor_sample(), condor_class())) 
        records.append(record(dolphin_sample(), dolphin_class())) 
        records.append(record(dragon_sample(), dragon_class())) 
    
    EPOCHS = 10 
    for e in range(0, EPOCHS): 
        print("epoch: %d" % e) 
        for t, c in records: 
            data = cv2.ml.TrainData_create(t, cv2.ml.ROW_SAMPLE, c) 
            if animals_net.isTrained(): 
                animals_net.train(data, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE) 
            else: 
                animals_net.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE) 
    
    TESTS = 100 
 
    dog_results = 0 
    for x in range(0, TESTS): 
        clas = int(animals_net.predict( 
            np.array([dog_sample()], np.float32))[0]) 
        print("class: %d" % clas) 
        if clas == 0: 
            dog_results += 1 
     
    condor_results = 0 
    for x in range(0, TESTS): 
        clas = int(animals_net.predict( 
            np.array([condor_sample()], np.float32))[0]) 
        print("class: %d" % clas) 
        if clas == 1: 
            condor_results += 1 
     
    dolphin_results = 0 
    for x in range(0, TESTS): 
        clas = int(animals_net.predict( 
            np.array([dolphin_sample()], np.float32))[0]) 
        print("class: %d" % clas) 
        if clas == 2: 
            dolphin_results += 1 
     
    dragon_results = 0 
    for x in range(0, TESTS): 
        clas = int(animals_net.predict( 
            np.array([dragon_sample()], np.float32))[0]) 
        print("class: %d" % clas) 
        if clas == 3: 
            dragon_results += 1 
            
    print("dog accuracy: %.2f%%" % (100.0 * dog_results / TESTS)) 
    print("condor accuracy: %.2f%%" % (100.0 * condor_results / TESTS)) 
    print("dolphin accuracy: %.2f%%" % (100.0 * dolphin_results / TESTS)) 
    print("dragon accuracy: %.2f%%" % (100.0 * dragon_results / TESTS)) 


def mnist_classifier():
    """
    mnist format:
    ((training_images, training_ids), 
    (test_images, test_ids))
    
    - training_images is a NumPy array of 60,000 images, where each image is a vector 
    of 784-pixel values (flattened from an original shape of 28 x 28 pixels). The 
    pixel values are floating-point numbers in the range 0.0 (black) to 1.0 (white), 
    inclusive.
    - training_ids is a NumPy array of 60,000 digit IDs, where each ID is a number 
    in the range 0 to 9, inclusive. training_ids[i] corresponds to training_images[i].
    - test_images is a NumPy array of 10,000 images, where each image is a vector 
    of 784-pixel values (flattened from an original shape of 28 x 28 pixels). 
    The pixel values are floating-point numbers in the range 0.0 (black) to 1.0 
    (white), inclusive.
    - test_ids is a NumPy array of 10,000 digit IDs, where each ID is a number 
    in the range 0 to 9, inclusive. test_ids[i] corresponds to test_images[i].
    """
    def load_data(): 
        mnist_path = '/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionImages/classifier_mnist/mnist.pkl'
        with open(mnist_path, 'rb') as f:
            training_data, test_data, _ = pickle.load(f, encoding='latin1') 
        return (training_data, test_data) 
    def vectorized_result(j): 
        e = np.zeros((10,), np.float32) 
        e[j] = 1.0 
        return e 
    def wrap_data():
        '''
        Must be a vector with 10 elements (for 10 classes of digits), rather than 
        a single digit ID.
        '''
        tr_d, te_d = load_data() 
        training_inputs = tr_d[0] 
        training_results = [vectorized_result(y) for y in tr_d[1]] 
        training_data = zip(training_inputs, training_results) 
        test_data = zip(te_d[0], te_d[1]) 
        return (training_data, test_data) 
    
    def create_ann(hidden_nodes=60): 
        ann = cv2.ml.ANN_MLP_create() 
        ann.setLayerSizes(np.array([784, hidden_nodes, 10])) 
        ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0) 
        ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1) 
        ann.setTermCriteria( 
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 
             100, 1.0)) 
        return ann 

    def train(ann, samples=50000, epochs=10): 
 
        tr, test = wrap_data() 
     
        # Convert iterator to list so that we can iterate multiple  
        # times in multiple epochs. 
        tr = list(tr) 
     
        for epoch in range(epochs): 
            print("Completed %d/%d epochs" % (epoch, epochs)) 
            counter = 0 
            for img in tr: 
                if (counter > samples): 
                    break 
                if (counter % 1000 == 0): 
                    print("Epoch %d: Trained on %d/%d samples" %
                           (epoch, counter, samples)) 
                counter += 1 
                sample, response = img 
                data = cv2.ml.TrainData_create( 
                    np.array([sample], dtype=np.float32), 
                    cv2.ml.ROW_SAMPLE, 
                    np.array([response], dtype=np.float32)) 
                if ann.isTrained(): 
                    ann.train(data, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE) 
                else: 
                    ann.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE) 
        print("Completed all epochs!") 
        return ann, test 
        
    def predict(ann, sample): 
        if sample.shape != (784,): 
            if sample.shape != (28, 28): 
                sample = cv2.resize(sample, (28, 28), 
                                interpolation=cv2.INTER_LINEAR) 
            sample = sample.reshape(784,) 
        return ann.predict(np.array([sample], dtype=np.float32)) 
 
    def test(ann, test_data): 
        num_tests = 0 
        num_correct = 0 
        for img in test_data: 
            num_tests += 1 
            sample, correct_digit_class = img 
            digit_class = predict(ann, sample)[0] 
            if digit_class == correct_digit_class: 
                num_correct += 1 
        print('Accuracy: %.2f%%' % (100.0 * num_correct / num_tests)) 

    ann, test_data = train(create_ann()) 
    test(ann, test_data) 
    

def main_mnist_classifier():
    
    def inside(r1, r2): 
        x1, y1, w1, h1 = r1 
        x2, y2, w2, h2 = r2 
        return ((x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and  
                (y1+h1 < y2+h2))
    
    '''To further ensure that the bounding rectangles meet the classifier's needs, 
    we will use another helper function, called wrap_digit, to convert a tightly-fitting 
    bounding rectangle into a square with padding around the digit.'''
    def wrap_digit(rect, img_w, img_h): 
        x, y, w, h = rect 
        x_center = x + w//2 
        y_center = y + h//2 
        if (h > w): 
            w = h 
            x = x_center - (w//2) 
        else: 
            h = w 
            y = y_center - (h//2) 
        padding = 5 
        x -= padding 
        y -= padding 
        w += 2 * padding 
        h += 2 * padding 
        '''To avoid out of bounds problems, we crop the rectangle so that it lies entirely 
        within the image. This could leave us with non-square rectangles in these edge cases, 
        but this is an acceptable compromise; we would prefer to use a non-square region 
        of interest rather than having to entirely throw out a detected digit just 
        because it is at the edge of the image.'''
        if x < 0: 
            x = 0 
        elif x > img_w: 
            x = img_w 
        
        if y < 0: 
            y = 0 
        elif y > img_h: 
            y = img_h 
        
        if x+w > img_w: 
            w = img_w - x 
        
        if y+h > img_h: 
            h = img_h - y 
        
        return x, y, w, h

    def load_data(): 
        mnist_path = '/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionImages/classifier_mnist/mnist.pkl'
        with open(mnist_path, 'rb') as f:
            training_data, test_data, _ = pickle.load(f, encoding='latin1') 
        return (training_data, test_data) 
    def vectorized_result(j): 
        e = np.zeros((10,), np.float32) 
        e[j] = 1.0 
        return e 
    def wrap_data():
        '''
        Must be a vector with 10 elements (for 10 classes of digits), rather than 
        a single digit ID.
        '''
        tr_d, te_d = load_data() 
        training_inputs = tr_d[0] 
        training_results = [vectorized_result(y) for y in tr_d[1]] 
        training_data = zip(training_inputs, training_results) 
        test_data = zip(te_d[0], te_d[1]) 
        return (training_data, test_data) 
    
    def create_ann(hidden_nodes=60): 
        ann = cv2.ml.ANN_MLP_create() 
        ann.setLayerSizes(np.array([784, hidden_nodes, 10]))
        """Also cv2.ml.ANN_MLP_IDENTITY, cv2.ml.ANN_MLP_GAUSSIAN, cv2.ml.ANN_MLP_RELU, 
        and cv2.ml.ANN_MLP_LEAKYRELU"""
        ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
        """Also cv2.ml.ANN_MLP_BACKPROP. The other options include cv2.ml.ANN_MLP_RPROP 
        and cv2.ml.ANN_MLP_ANNEAL"""
        ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1) 
        ann.setTermCriteria( 
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 
             100, 1.0)) 
        return ann 

    def train(ann, samples=50000, epochs=10): 
 
        tr, test = wrap_data() 
     
        # Convert iterator to list so that we can iterate multiple  
        # times in multiple epochs. 
        tr = list(tr) 
     
        for epoch in range(epochs): 
            print("Completed %d/%d epochs" % (epoch, epochs)) 
            counter = 0 
            for img in tr: 
                if (counter > samples): 
                    break 
                if (counter % 1000 == 0): 
                    print("Epoch %d: Trained on %d/%d samples" %
                           (epoch, counter, samples)) 
                counter += 1 
                sample, response = img 
                data = cv2.ml.TrainData_create( 
                    np.array([sample], dtype=np.float32), 
                    cv2.ml.ROW_SAMPLE, 
                    np.array([response], dtype=np.float32)) 
                if ann.isTrained(): 
                    ann.train(data, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE) 
                else: 
                    ann.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE) 
        print("Completed all epochs!") 
        return ann, test 
        
    def predict(ann, sample): 
        if sample.shape != (784,): 
            if sample.shape != (28, 28): 
                sample = cv2.resize(sample, (28, 28), 
                                interpolation=cv2.INTER_LINEAR) 
            sample = sample.reshape(784,) 
        return ann.predict(np.array([sample], dtype=np.float32)) 
 
    def test(ann, test_data): 
        num_tests = 0 
        num_correct = 0 
        for img in test_data: 
            num_tests += 1 
            sample, correct_digit_class = img 
            digit_class = predict(ann, sample)[0] 
            if digit_class == correct_digit_class: 
                num_correct += 1 
        print('Accuracy: %.2f%%' % (100.0 * num_correct / num_tests)) 

    ann, test_data = train( 
        create_ann(60), 50000, 10)
    
    img_path = "./digit_images/digits_0.jpg" 
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    '''convert the image into grayscale and blur it in order to remove noise and 
    make the darkness of the ink more uniform.'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    cv2.GaussianBlur(gray, (7, 7), 0, gray) 
    '''Apply a threshold and some morphology operations to ensure that the numbers 
    stand out from the background and that the contours are relatively free of 
    irregularities, which might throw off the prediction.'''
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) 
    erode_kernel = np.ones((2, 2), np.uint8) 
    thresh = cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    '''Note the threshold flag, cv2.THRESH_BINARY_INV, which is for an inverse 
    binary threshold. Since the samples in the MNIST database are white on black 
    (and not black on white), we turn the image into a black background with white 
    numbers. We use the thresholded image for both detection and classification.'''
    '''Need to separately detect each digit in the picture. As a step toward this, 
    first, we need to find the contours'''
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    '''Then, we iterate through the contours and find their bounding rectangles. 
    We discard any rectangles that we deem too large or too small to be digits. 
    We also discard any rectangles that are entirely contained in other rectangles. 
    The remaining rectangles are appended to a list of good rectangles, which 
    (we believe) contain individual digits.'''
    rectangles = [] 
 
    img_h, img_w = img.shape[:2] 
    img_area = img_w * img_h 
    for c in contours: 
     
        a = cv2.contourArea(c) 
        if a >= 0.98 * img_area or a <= 0.0001 * img_area: 
            continue 
     
        r = cv2.boundingRect(c) 
        is_inside = False 
        for q in rectangles: 
            if inside(r, q): 
                is_inside = True 
                break 
        if not is_inside: 
            rectangles.append(r) 
    
    for r in rectangles: 
        x, y, w, h = wrap_digit(r, img_w, img_h) 
        roi = thresh[y:y+h, x:x+w] 
        digit_class = int(predict(ann, roi)[0]) 
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) 
        cv2.putText(img, "%d" % digit_class, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("thresh", thresh) 
    cv2.imshow("detected and classified digits", img) 
    cv2.waitKey() 

def optical_character_recognition_OCR():
    pass


"""
caffe_model = cv2.dnn.readNetFromCaffe( 
    'my_model_description.protext', 'my_model.caffemodel') 
 
tensor_flow_model = cv2.dnn.readNetFromTensorflow( 
    'my_model.pb', 'my_model_description.pbtxt') 
 
# Some Torch models use the .t7 extension and others use 
# the .net extension. 
torch_model_0 = cv2.dnn.readNetFromTorch('my_model.t7') 
torch_model_1 = cv2.dnn.readNetFromTorch('my_model.net') 
 
darknet_model = cv2.dnn.readNetFromDarket( 
    'my_model_description.cfg', 'my_model.weights') 
 
onnx_model = cv2.dnn.readNetFromONNX('my_model.onnx') 
 
dldt_model = cv2.dnn.readNetFromModelOptimizer( 
    'my_model_description.xml', 'my_model.bin')

After we load a model, we need to preprocess the data we will use with the model. 
The necessary preprocessing is specific to the way the given DNN was designed and 
trained, so any time we use a third-party DNN, we must read about how that DNN was 
designed and trained. OpenCV provides a function, cv2.dnn.blobFromImage, that can 
perform some common preprocessing steps, depending on the parameters we pass to it. 
We can perform other preprocessing steps manually before passing data to this function.

A neural network's input vector is sometimes called a tensor or blob – hence the 
function's name, cv2.dnn.blobFromImage.
"""

"""Before delving into the code, let's introduce the DNN that we will use. It is a 
Caffe version of a model called MobileNet-SSD, which uses a hybrid of a framework 
from Google called MobileNet and another framework called Single Shot Detector (SSD) 
MultiBox. The latter framework has a GitHub repository at 
https://github.com/weiliu89/caffe/tree/ssd/. The training technique for the Caffe 
version of MobileNet-SSD is provided by a project on GitHub at 
https://github.com/chuanqi305/MobileNet-SSD/.

MobileNetSSD_deploy.caffemodel: This is the model.
MobileNetSSD_deploy.prototxt: This is the text file that describes the model's parameters.
"""
def coffee_dnn_apply():
    model = cv2.dnn.readNetFromCaffe( 
        'objects_data/MobileNetSSD_deploy.prototxt', 
        'objects_data/MobileNetSSD_deploy.caffemodel') 
    """We need to define some preprocessing parameters that are specific to this model. 
    It expects the input image to be 300 pixels high. Also, it expects the pixel values 
    in the image to be on a scale from -1.0 to 1.0. This means that, relative to 
    the usual scale from 0 to 255, it is necessary to subtract 127.5 and then divide 
    by 127.5. We define the parameters as follows:"""
    blob_height = 300 
    color_scale = 1.0/127.5 
    average_color = (127.5, 127.5, 127.5) 
    """We also define a confidence threshold, representing the minimum confidence 
    score that we require in order to accept a detection as a real object:"""
    confidence_threshold = 0.5
    labels = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
    'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 
    'horse', 'motorbike', 'person', 'potted plant', 'sheep', 
    'sofa', 'train', 'TV or monitor']
    """Later, when we use class IDs to look up labels in our list, we must remember 
    to subtract 1 from the ID in order to obtain an index in the range 0 to 19 
    (not 1 to 20)."""
    """For each frame, we begin by calculating the aspect ratio. Remember that this 
    DNN expects the input to be based on an image that is 300 pixels high; however, 
    the width can vary in order to match the original aspect ratio. The following 
    code snippet shows how we capture a frame and calculate the appropriate input size:"""
    cap = cv2.VideoCapture(0) 
    success, frame = cap.read() 
    while success: 
        h, w = frame.shape[:2] 
        aspect_ratio = w/h 
        # Detect objects in the frame. 
        blob_width = int(blob_height * aspect_ratio) 
        blob_size = (blob_width, blob_height)
        """At this point, we can simply use the cv2.dnn.blobFromImage function, 
        with several of its optional arguments, to perform the necessary preprocessing, 
        including resizing the frame and converting its pixel data into a scale 
        from -1.0 to 1.0:"""
        blob = cv2.dnn.blobFromImage( 
            frame, scalefactor=color_scale, size=blob_size, 
            mean=average_color) 
        model.setInput(blob) 
        results = model.forward()
        """The results are an array, in a format that is specific to the model 
        we are using.
        For this object detection DNN – and for other DNNs trained with the SSD 
        framework – the results include a subarray of detected objects, each with 
        its own confidence score, rectangle coordinates, and class ID. The 
        following code shows how to access these, as well as how to use an ID to 
        look up a label in the list we defined earlier:"""
        # Iterate over the detected objects. 
        for object in results[0, 0]: 
            confidence = object[2] 
            if confidence > confidence_threshold: 
        
                # Get the object's coordinates. 
                x0, y0, x1, y1 = (object[3:7] * [w, h, w, h]).astype(int) 
        
                # Get the classification result. 
                id = int(object[1]) 
                label = labels[id - 1]
                # Draw a blue rectangle around the object. 
                cv2.rectangle(frame, (x0, y0), (x1, y1), 
                              (255, 0, 0), 2) 
            
                # Draw the classification result and confidence. 
                text = '%s (%.1f%%)' % (label, confidence * 100.0) 
                cv2.putText(frame, text, (x0, y0 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        cv2.imshow('Objects', frame) 
        k = cv2.waitKey(1) 
        if k == 27: # Escape 
            break 
        success, frame = cap.read() 

"""
For this demonstration, we are going to use one DNN to detect faces and two other 
DNNs to classify the age and gender of each detected face. Specifically, we will 
use pre-trained Caffe models that are stored in the following files in the 
faces_data folder.

Here is an inventory of the files in this folder, and of the files' origins:

detection/res10_300x300_ssd_iter_140000.caffemodel: This is the DNN for face 
detection. The OpenCV team has provided this file at 
https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel. 
This Caffe model was trained with the SSD framework 
(https://github.com/weiliu89/caffe/tree/ssd/). Thus, its topology is similar to 
the MobileNet-SSD model that we used in the previous section's example.

detection/deploy.prototxt: This is the text file that describes the parameters 
of the preceding DNN for face detection. The OpenCV team provides this file at 
https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt.

The chapter10/faces_data/age_gender_classification folder contains the following 
files, which are all provided by Gil Levi and Tal Hassner in their GitHub repository 
(https://github.com/GilLevi/AgeGenderDeepLearning/) and on their project page 
(https://talhassner.github.io/home/publication/2015_CVPR) for their work on age 
and gender classification:

- age_net.caffemodel: This is the DNN for age classification.
- age_net_deploy.protext: This is the text file that describes the parameters of 
the preceding DNN for age classification.
- gender_net.caffemodel: This is the DNN for gender classification.
- gender_net_deploy.protext: This is the text file that describes the parameters 
of the preceding DNN for age classification.
- average_face.npy and average_face.png: These files represent the average faces 
in the classifiers' training dataset. The original file from Levi and Hassner is 
called mean.binaryproto, but we have converted it into a NumPy-readable format 
and a standard image format, which are more convenient for our purposes.
"""
def face_dnn_apply():
    face_model = cv2.dnn.readNetFromCaffe( 
        'faces_data/detection/deploy.prototxt', 
        'faces_data/detection/res10_300x300_ssd_iter_140000.caffemodel'
        ) 
    face_blob_height = 300 
    face_average_color = (104, 177, 123) 
    face_confidence_threshold = 0.995
    '''We do not need to define labels for this DNN because it does not perform 
    any classification; it just predicts the coordinates of face rectangles.'''
    age_model = cv2.dnn.readNetFromCaffe( 
       'faces_data/age_gender_classification/age_net_deploy.prototxt', 
        'faces_data/age_gender_classification/age_net.caffemodel') 
    age_labels = ['0-2', '4-6', '8-12', '15-20', 
                  '25-32', '38-43', '48-53', '60+']
    '''Note that in this model, the age labels have gaps between them. For example, 
    '0-2' is followed by '4-6'. Thus, if a person is actually 3 years old, the 
    classifier has no proper label for this case; at best, it can pick either of 
    the neighboring ranges, '0-2' or '4-6'. Presumably, the model's authors 
    deliberately chose disconnected ranges, in an effort to ensure that the classes 
    are separable with respect to the inputs. Let's consider the alternative. 
    Based on data from facial images, is it possible to separate a group of people 
    who are 4 years old from a group of people who are 4-years-less-a-day? Surely 
    it isn't; they look the same. Thus, it would be wrong to formulate a classification 
    problem based on contiguous age ranges. A DNN could be trained to predict age 
    as a continuous variable (such as a floating-point number of years), but this 
    would be altogether different than a classifier, which predicts confidence 
    scores for various classes.'''
    gender_model = cv2.dnn.readNetFromCaffe( 
        'faces_data/age_gender_classification/gender_net_deploy.prototxt', 
        'faces_data/age_gender_classification/gender_net.caffemodel') 
    gender_labels = ['male', 'female']
    '''The age and gender classifiers use the same blob size and the same average. 
    Rather than using a single color as the average, they use an average facial 
    image, which we will load (as a NumPy array in floating-point format) from an 
    NPY file. Later, we will subtract this average facial image from an actual 
    facial image before we perform classification. Here are the definitions of 
    the blob size and average image:'''
    age_gender_blob_size = (256, 256) 
    age_gender_average_image = np.load( 
        'faces_data/age_gender_classification/average_face.npy') 
    cap = cv2.VideoCapture(0) 
 
    success, frame = cap.read() 
    while success: 
     
        h, w = frame.shape[:2] 
        aspect_ratio = w/h 
     
        # Detect faces in the frame. 
     
        face_blob_width = int(face_blob_height * aspect_ratio) 
        face_blob_size = (face_blob_width, face_blob_height) 
     
        face_blob = cv2.dnn.blobFromImage( 
            frame, size=face_blob_size, mean=face_average_color) 
     
        face_model.setInput(face_blob) 
        face_results = face_model.forward() 
        
        # Iterate over the detected faces. 
        for face in face_results[0, 0]: 
            face_confidence = face[2] 
            if face_confidence > face_confidence_threshold: 
        
                # Get the face coordinates. 
                x0, y0, x1, y1 = (face[3:7] * [w, h, w, h]).astype(int)
                
                # Classify the age and gender of the face based on a 
                # square region of interest that includes the neck. 
                
                y1_roi = y0 + int(1.2*(y1-y0)) 
                x_margin = ((y1_roi-y0) - (x1-x0)) // 2 
                x0_roi = x0 - x_margin 
                x1_roi = x1 + x_margin 
                if x0_roi < 0 or x1_roi > w or y0 < 0 or y1_roi > h: 
                    # The region of interest is partly outside the 
                    # frame. Skip this face. 
                    continue 
                
                age_gender_roi = frame[y0:y1_roi, x0_roi:x1_roi] 
                scaled_age_gender_roi = cv2.resize( 
                    age_gender_roi, age_gender_blob_size, 
                    interpolation=cv2.INTER_LINEAR).astype(np.float32) 
                scaled_age_gender_roi[:] -= age_gender_average_image 
                age_gender_blob = cv2.dnn.blobFromImage( 
                    scaled_age_gender_roi, size=age_gender_blob_size) 

            age_model.setInput(age_gender_blob) 
            age_results = age_model.forward() 
            age_id = np.argmax(age_results) 
            age_label = age_labels[age_id] 
            age_confidence = age_results[0, age_id] 
            
            gender_model.setInput(age_gender_blob) 
            gender_results = gender_model.forward() 
            gender_id = np.argmax(gender_results) 
            gender_label = gender_labels[gender_id] 
            gender_confidence = gender_results[0, gender_id] 
            
            # Draw a blue rectangle around the face. 
            cv2.rectangle(frame, (x0, y0), (x1, y1), 
                              (255, 0, 0), 2) 
            
            # Draw a yellow square around the region of interest 
            # for age and gender classification. 
            cv2.rectangle(frame, (x0_roi, y0), (x1_roi, y1_roi), 
                              (0, 255, 255), 2) 
            
            # Draw the age and gender classification results. 
            text = '%s years (%.1f%%), %s (%.1f%%)' % ( 
                    age_label, age_confidence * 100.0, 
                    gender_label, gender_confidence * 100.0) 
            cv2.putText(frame, text, (x0_roi, y0 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
            
        cv2.imshow('Faces, age, and gender', frame) 
 
        k = cv2.waitKey(1) 
        if k == 27: # Escape 
            break 
         
        success, frame = cap.read() 

"""Let’s take a look at how we can use MediaPipe, OpenCV and Tensorflow to build 
a ML classifier that will recognize a number of hand gestures such as an open 
hand, an “OK” sign, a “peace” sign and a “thumb up” sign."""
"""
Detecting poses, faces, hands and gestures => MediaPipe
MediaPipe is a library developed by Google that makes the detection of body poses, 
faces, hands, iris and lots more (to do with a human body) a trivial task. It is 
highly optimized and performs really well even on commodity hardware like normal 
development machines.
"""
def hands_dnn_apply():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()

"""
The two biggest challenges are the generation of data (it’s a custom gesture 
recognition classifier, we presume such data does not exist at all) and creating 
a deep learning model that’s accurate enough (upwards of 98%).

Generating Labeled Training Data
Let’s tackle the first challenge by creating data. I am going to offer my approach 
to this but clearly you can modify it to suit your needs.
The approach I offer is as follows: start capturing video input, and for as long 
as you keep pressing one number key (0 to 3), the landmark coordinates (normalized) 
will be stored in a csv file with a label corresponding to the number key you are 
pressing.
To increase the model accuracy, make sure to move the hand around, tilt it and 
turn it, all while keeping the gesture “intact” so to speak. Even more importantly, 
don’t break the gesture or you will be polluting the training dataset and introducing 
noise. Lastly, make sure to be consistent with the gestures and keys, don’t mix 
number keys and gestures up.
"""
def gesture_data_generation():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    def main():
        # For webcam input:
        cap = cv2.VideoCapture(0)
        number = 0
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                receivedKey = cv2.waitKey(20)
                number = receivedKey - 48
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                '''I tailored the script to train only 4 classes, so I spare the 
                program from performing any hand detection unless we are generating 
                training data, which means that the number variable needs to be a 
                value between 0 and 3'''
                if results.multi_hand_landmarks and number in [0, 1, 2, 3]:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = u.calc_landmark_list(image, hand_landmarks)
                        pre_processed_landmark_list = u.pre_process_landmark(
                            landmark_list)
                        log_csv(number, pre_processed_landmark_list)
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                final = cv2.flip(image, 1)
                text = ""
                if number == -1:
                    text = "Press key for gesture number"
                else:
                    text = "Gesture: {}".format(number)
                cv2.putText(final, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', final)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
    main()
        
def log_csv(number, landmark_list):
    if number > 9 or number == -1:
        pass
    csv_path = csv_path = 'model/keypoint_classifier/keypoint_yann.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
    return

def classifier_gesture():
    dataset = 'model/keypoint_classifier/keypoint.csv'
    model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
    tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
    NUM_CLASSES = 4
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    # Then we load the labels:
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    # As expected, we only load the column at index 0, which we skipped for the training data.
    RANDOM_SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.summary()
    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    # And then we compile the model
    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Now we “fit” the data (we train the model):
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
    model.save(model_save_path, include_optimizer=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    open(tflite_save_path, 'wb').write(tflite_quantized_model)
    
    class KeyPointClassifier(object):
        def __init__(
            self,
            model_path='model/keypoint_classifier/keypoint_classifier.tflite',
            num_threads=1,
        ):
            self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                                   num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        def __call__(
            self,
            landmark_list,
        ):
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            self.interpreter.invoke()
            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
            result_index = np.argmax(np.squeeze(result))
            return result_index
        
    # TFLite DOCUMENTATION
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    kpclf = KeyPointClassifier()
    gestures = {
        0: "Open Hand",
        1: "Thumb up",
        2: "OK",
        3: "Peace",
        4: "No Hand Detected"
    }
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gesture_index = 4
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = u.calc_landmark_list(image, hand_landmarks)
                    keypoints = u.pre_process_landmark(landmark_list)
                    gesture_index = kpclf(keypoints)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            final = cv2.flip(image, 1)
            cv2.putText(final, gestures[gesture_index],
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            cv2.imshow('MediaPipe Hands', final)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    # Note these lines:
    kpclf = KeyPointClassifier()
    gestures = {
        0: "Open Hand",
        1: "Thumb up",
        2: "OK",
        3: "Peace",
        4: "No Hand Detected"
    }

if __name__=='__main__':
    # ann_sample()
    # animal_classification_ann()
    # mnist_classifier()
    # main_mnist_classifier()
    # coffee_dnn_apply()
    # face_dnn_apply()
    # hands_dnn_apply()
    # gesture_data_generation()
    classifier_gesture()