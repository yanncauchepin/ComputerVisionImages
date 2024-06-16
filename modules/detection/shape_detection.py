import numpy as np
import cv2


class ShapeDetection():
    
    
    """The Hough transformation.
    We can do this with either the HoughLines function or the HoughLinesP function. 
    The former uses the standard Hough transform, while the latter uses the probabilistic 
    Hough transform (hence the P in the name). The probabilistic version is so-called 
    because it only analyzes a subset of the image's points and estimates the probability 
    that these points all belong to the same line. This implementation is an optimized 
    version of the standard Hough transform; it is less computationally intensive and 
    executes faster. HoughLinesP is implemented so that it returns the two endpoints 
    of each detected line segment, whereas HoughLines is implemented so that it returns 
    a representation of each line as a single point and an angle, without information 
    about endpoints."""
    
    
    def __init__(self):
        pass


    '''
    The parameters of HoughLinesP are as follows:
    * The binary image from Canny or another edge detection filter.
    * The resolution or step size to use when searching for lines. rho is the positional 
    step size in pixels, while theta is the rotational step size in radians. For example, 
    if we specify rho=1 and theta=np.pi/180.0, we search for lines that are separated 
    by as little as 1 pixel and 1 degree.
    * The threshold, which represents the threshold below which a line is discarded. 
    The Hough transform works with a system of bins and votes, with each bin representing 
    a line, so if a candidate line has at least the threshold number of votes, it is 
    retained; otherwise, it is discarded.
    * An optional argument, lines, which we do not use here and which is not really 
    useful in the function’s Python version. The lines argument can be used to provide 
    a list or array in which HoughLinesP will put the resulting lines; otherwise, it 
    will return a new list.
    * minLineLength and maxLineGap, which we mentioned previously.
    '''
    @staticmethod
    def houghlines_detection(image, show=True):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 120)
        '''Canny is not a strict requirement, but an image that has been denoised and 
        only represents edges is the ideal source for a Hough transform, so you will 
        find this to be a common practice.'''
        lines = cv2.HoughLinesP(edges, rho=1,
                                theta=np.pi/180.0,
                                threshold=20,
                                minLineLength=40,
                                maxLineGap=5)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show:
            cv2.imshow("Edges image", edges)
            cv2.imshow("Lines image", image)
            cv2.waitKey(0)
        return image


    '''OpenCV also has a function for detecting circles, called HoughCircles. It works 
    in a very similar fashion to HoughLines, but where minLineLength and maxLineGap 
    were the parameters used to discard or retain lines, HoughCircles instead lets us 
    specify a minimum distance between a circle's centers, as well as minimum and maximum 
    values for a circle's radius.'''
    @staticmethod
    def houghcircles_detection(image, show=True):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 5)
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT,
                                   1,120, param1=100, param2=30,
                                   minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2],
                       (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image,(i[0], i[1]), 2,
                       (0, 0, 255), 3)
        if show:
            cv2.imshow("Cirlces image", image)
            cv2.waitKey(0)
        return image
    '''Note that we do not call the Canny function here because, when we call HoughCircles 
    with the cv2.HOUGH_GRADIENT option, the latter function already applies the Canny 
    algorithm internally, using the value of param1 as the first Canny threshold and 
    half this value as the second Canny threshold. Also, with cv2.HOUGH_GRADIENT, the 
    value of param2 is a threshold for weeding out overlapping circle detections; a 
    higher value leads to fewer overlapping detections.'''


    '''OpenCV's implementations of the Hough transform are limited to detecting lines 
    and circles; however, we already implicitly explored shape detection in general 
    when we talked about approxPolyDP. This function allows for the approximation of 
    polygons, so if your image contains polygons, they will be accurately detected 
    through the combined use of cv2.findContours and cv2.approxPolyDP.'''
