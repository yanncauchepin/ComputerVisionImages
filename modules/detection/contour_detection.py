import numpy as np
import cv2


class ContourDetection():
    
    
    def __init__(self):
        pass


    @staticmethod
    def contours(image, show=True):
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.drawContours(color, contours, -1, (0,255,0), 2)
        if show:
            cv2.imshow("Contours image", image)
        return image


    @staticmethod
    def bounding_shape(image, show=True):
        image = cv2.pyrDown(image)
        ret, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                    127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # Bounding box
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
            # Minimum enclosing rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect) # calculate float coordinates of the minimum area rectangle     
            box = np.int0(box) # normalize coordinates to integers
            cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
            # Minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            image = cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
        if show:
            cv2.imshow("Bounding shape image", image)
        return image
    

    '''A convex shape is one where there are no two points within this shape whose 
    connecting line goes outside the perimeter of the shape itself.
    
    The first facility that OpenCV offers to calculate the approximate bounding 
    polygon of a shape is cv2.approxPolyDP. The DP in the function name stands for 
    the Douglas-Peucker algorithm. This function takes three parameters:
    * A contour.
    * An epsilon value representing the maximum discrepancy between the original contour 
    and the approximated polygon (the lower the value, the closer the approximated 
    value will be to the original contour).
    * A Boolean flag. If it is True, it signifies that the polygon is closed.
    The epsilon value is of vital importance to obtain a useful contour, so let's 
    understand what it represents. Epsilon is the maximum difference between the 
    approximated polygon's perimeter and the original contour's perimeter. The smaller 
    this difference is, the more the approximated polygon will be similar to the original 
    contour.
    Now that we know what an epsilon is, we need to obtain contour perimeter information 
    as a reference value. This can be obtained with the cv2.arcLength function of OpenCV.
    OpenCV also offers a cv2.convexHull function for obtaining processed contour 
    information for convex shapes.'''
    @staticmethod
    def convex_contours(image, show=True):
        image = cv2.pyrDown(image)
        ret, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                    127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        black = np.zeros_like(image)
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            '''DP for the Douglas-Peucker algorithm'''
            hull = cv2.convexHull(cnt)
            cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
            cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)
        if show:
            cv2.imshow("Hull detection image", black)
        return black