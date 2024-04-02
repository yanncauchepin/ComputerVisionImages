import os
import numpy as np
import cv2
from scipy import ndimage

"""EDGE DETECTION"""


'''numpy has a fast Fourier transform (FFT) package which contains fft2 method 
which allow to compute a discrete Forurier transform (DFT) of the Image.
Fourier transform -> find the magnitude spectrum of an image taht represents the 
original image in terms of its changes.
Fourier transform -> common algorithms for image processing
Fourier transform -> replaced by simpler function that process a small region or
neighborhood.
'''

"""HPF = filter that examine a region + boost the intensity of surrounding pixels.
kernel = set of weights applied on region pixel to generate a single pixel.
kernel = convolution matrix
example = [[ 0,     -0.25,  0   ], [ -0.25,   1,    -0.25], [  0,     -0.25,  0   ]]
-> this example lead to get the average difference with immediate horizontal neighbors.
-> this example represents a so-called high-boost filter, a type of HPF, and is 
particularly effective in edge detection.
Edge detection kernel -> typically sum up to 0."""

def custom_kernel_HPF(*, kernel, image):
    if kernel.sum() != 0:
        raise Exception('Kernel sum up is not null.')
    kernel_image = ndimage.convolve(image, kernel)
    '''OpenCV provides a filter2D function (for convolution with 2D arrays) and 
    a sepFilter2D function (for the special case of a 2D kernel that can be decomposed 
    into two one-dimensional kernels).'''
    # cv2.filter2D(src, -1, kernel, dst)
    '''The second argument specifies the per-channel depth of the destination image 
    (such as cv2.CV_8U for 8 bits per channel). A negative value (such as to one 
    being used here) means that the destination image has the same depth as the 
    source image.
    For color images, note that filter2D applies the kernel equally to each channel. 
    To use different kernels on different channels, we would also have to use OpenCV’s 
    split and merge functions.
    '''
    cv2.imshow('Custom kernel image', kernel_image)


'''LPF = filter that smooth or flatten the pixel intensity if the difference from
surroundin pixels is lower than a certain threshold.
-> used in denoising and blurring.
Gaussian blur low-pass kernel -> most popular blurring/smoothing filters
-> attenuates the intensity of high-frequency signals.'''

def gaussian_blur_low_pass_kernel(*, size_kernel, image, difference=True):
    blur_kernel_image = cv2.GaussianBlur(image, (size_kernel, size_kernel), 0)
    cv2.imshow('Gaussian blur low-pass kernel image', blur_kernel_image)
    if difference:
        cv2.imshow('Differential high-pass kernel image', image - blur_kernel_image)


'''OpenCV provides many edge-finding filters, including Laplacian, Sobel, and Scharr. 
These filters are supposed to turn non-edge regions into black and turn edge regions 
into white or saturated colors. However, they are prone to misidentifying noise as edges. 
This flaw can be mitigated by blurring an image before trying to find its edges. 
OpenCV also provides many blurring filters, including blur (a simple average), 
medianBlur, and GaussianBlur. The arguments for the edge-finding and blurring filters 
vary but always include ksize, an odd whole number that represents the width and 
height (in pixels) of a filter's kernel.

For blurring, let's use medianBlur, which is effective in removing digital video 
noise, especially in color images. For edge-finding, let's use Laplacian, which 
produces bold edge lines, especially in grayscale images. After applying medianBlur, 
but before applying Laplacian, we should convert the image from BGR into grayscale.

Once we have the result of Laplacian, we can invert it to get black edges on a white 
background. Then, we can normalize it (so that its values range from 0 to 1) and 
then multiply it with the source image to darken the edges.'''


'''The Canny edge detection algorithm is complex but also quite interesting. 
It is a five-step process:
1. Denoise the image with a Gaussian filter.
2. Calculate the gradients.
3. Apply non-maximum suppression (NMS) on the edges. Basically, this means that 
the algorithm selects the best edges from a set of overlapping edges.
4. Apply a double threshold to all the detected edges to eliminate any false positives.
5. Analyze all the edges and their connection to each other to keep the real edges 
and discard the weak ones.'''
def canny_kernel(*, width_kernel, height_kernel, image):
    canny_kernel_image = cv2.Canny(image, width_kernel, height_kernel)
    cv2.imshow('Canny kernel', canny_kernel_image)

'''After finding Canny edges, we can do further analysis of the edges in order to 
determine whether they match a common shape, such as a line or a circle. The Hough 
transform is one algorithm that uses Canny edges in this way.'''


"""CONTOUR DETECTION"""
    

def contours(*, image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
    cv2.imshow("Contours image", color)


def bounding_shape(*, image):
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
    cv2.imshow("Bounding shape image", image)

'''A convex shape is one where there are no two points within this shape whose 
connecting line goes outside the perimeter of the shape itself.'''

'''The first facility that OpenCV offers to calculate the approximate bounding 
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
information for convex shapes.
'''

def convex_contours(*, path_to_image):
    image = cv2.pyrDown(cv2.imread(path_to_image))
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
    cv2.imshow("Hull detection image", black)


"""DETECTING LINES, CIRCLES, SHAPES"""

'''The Hough transformation.
We can do this with either the HoughLines function or the HoughLinesP function. 
The former uses the standard Hough transform, while the latter uses the probabilistic 
Hough transform (hence the P in the name). The probabilistic version is so-called 
because it only analyzes a subset of the image's points and estimates the probability 
that these points all belong to the same line. This implementation is an optimized 
version of the standard Hough transform; it is less computationally intensive and 
executes faster. HoughLinesP is implemented so that it returns the two endpoints 
of each detected line segment, whereas HoughLines is implemented so that it returns 
a representation of each line as a single point and an angle, without information 
about endpoints.

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
def houghlines_detection(*, image):
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
    cv2.imshow("Edges image", edges)
    cv2.imshow("Lines image", image)


'''OpenCV also has a function for detecting circles, called HoughCircles. It works 
in a very similar fashion to HoughLines, but where minLineLength and maxLineGap 
were the parameters used to discard or retain lines, HoughCircles instead lets us 
specify a minimum distance between a circle's centers, as well as minimum and maximum 
values for a circle's radius.'''
def houghcircles_detection(*, image):
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
    cv2.imshow("Cirlces image", image)
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

if __name__=='__main__':
    path_to_image = 'Thoune3.jpg'
    image = cv2.imread(path_to_image)
    gray_image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    kernel_3 = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])
    kernel_5 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 1, 2, 1, -1],
                         [-1, 2, 4, 2, -1],
                         [-1, 1, 2, 1, -1],
                         [-1, -1, -1, -1, -1]])
    #custom_kernel_HPF(kernel=kernel_3, image=gray_image)
    #gaussian_blur_low_pass_kernel(size_kernel=17, image=gray_image, difference=True)
    #canny_kernel(width_kernel=200, height_kernel=300, image=gray_image)
    #contours_example()
    #contours(image=gray_image)
    #bounding_shape(image=bgr_image)
    #convex_contours(path_to_image=path_to_image)
    #houghlines_detection(image=image)
    houghcircles_detection(image=image)