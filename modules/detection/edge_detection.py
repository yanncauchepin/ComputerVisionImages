import cv2
from scipy import ndimage


class EdgeDetection():


    '''numpy has a fast Fourier transform (FFT) package which contains fft2 method 
    which allow to compute a discrete Fourier transform (DFT) of the Image.
    Fourier transform -> find the magnitude spectrum of an image that represents the 
    original image in terms of its changes.
    Fourier transform -> common algorithms for image processing
    Fourier transform -> replaced by simpler function that process a small region or
    neighborhood.
    '''
    

    def __init__(self):
        pass

    
    """HPF = filter that examine a region + boost the intensity of surrounding pixels.
    kernel = set of weights applied on region pixel to generate a single pixel.
    kernel = convolution matrix
    example = [[ 0,     -0.25,  0   ], [ -0.25,   1,    -0.25], [  0,     -0.25,  0   ]]
    -> this example lead to get the average difference with immediate horizontal neighbors.
    -> this example represents a so-called high-boost filter, a type of HPF, and is 
    particularly effective in edge detection.
    Edge detection kernel -> typically sum up to 0."""
    @staticmethod
    def custom_kernel_HPF(kernel, image, show=True):
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
        if show:
            cv2.imshow('Custom kernel image', kernel_image)
        return kernel_image


    '''LPF = filter that smooth or flatten the pixel intensity if the difference from
    surroundin pixels is lower than a certain threshold.
    -> used in denoising and blurring.
    Gaussian blur low-pass kernel -> most popular blurring/smoothing filters
    -> attenuates the intensity of high-frequency signals.'''
    @staticmethod
    def gaussian_blur_low_pass_kernel(size_kernel, image, difference=True, show=True):
        blur_kernel_image = cv2.GaussianBlur(image, (size_kernel, size_kernel), 0)
        if difference:
            difference_image = image - blur_kernel_image
        if show:
            cv2.imshow('Gaussian blur low-pass kernel image', blur_kernel_image)
            if difference:
                cv2.imshow('Differential high-pass kernel image', difference_image)
        if difference:
            return blur_kernel_image, difference_image
        else:
            return blur_kernel_image
        


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
    @staticmethod
    def canny_kernel(*, width_kernel, height_kernel, image):
        canny_kernel_image = cv2.Canny(image, width_kernel, height_kernel)
        cv2.imshow('Canny kernel', canny_kernel_image)
    '''After finding Canny edges, we can do further analysis of the edges in order to 
    determine whether they match a common shape, such as a line or a circle. The Hough 
    transform is one algorithm that uses Canny edges in this way.'''