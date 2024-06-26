#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:56:06 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   EDGE DETECTION
# =============================================================================

from ComputerVisionImages.modules.detection.edge_detection import EdgeDetection
class TestEdgeDetection():

    @staticmethod
    def test_custom_kernel_HPF_thoune_3():
        kernel_3 = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        EdgeDetection.custom_kernel_HPF(kernel=kernel_3, image=image_thoune_gray)
    
    @staticmethod
    def test_custom_kernel_HPF_thoune_5():
        kernel_5 = np.array([[-1, -1, -1, -1, -1],
                              [-1, 1, 2, 1, -1],
                              [-1, 2, 4, 2, -1],
                              [-1, 1, 2, 1, -1],
                              [-1, -1, -1, -1, -1]])
        EdgeDetection.custom_kernel_HPF(kernel=kernel_5, image=image_thoune_gray)
        
    @staticmethod
    def test_gaussian_blur_low_pass_kernel_thoune_17():
        EdgeDetection.gaussian_blur_low_pass_kernel(
            size_kernel=17, image=image_thoune_gray, difference=True)
        
    @staticmethod
    def test_gaussian_blur_low_pass_kernel_champex_17():
        EdgeDetection.gaussian_blur_low_pass_kernel(
            size_kernel=17, image=image_champex_gray, difference=True)

    @staticmethod 
    def test_canny_kernel_thoune_200():
        EdgeDetection.canny_kernel(
            width_kernel=200, height_kernel=200, image=image_thoune_gray)
        
    @staticmethod 
    def test_canny_kernel_champex_150():
        EdgeDetection.canny_kernel(
            width_kernel=150, height_kernel=150, image=image_champex_gray)

# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestEdgeDetection.test_custom_kernel_HPF_thoune_3()
    # TestEdgeDetection.test_custom_kernel_HPF_thoune_5()
    # TestEdgeDetection.test_gaussian_blur_low_pass_kernel_thoune_17()
    # TestEdgeDetection.test_gaussian_blur_low_pass_kernel_champex_17()
    # TestEdgeDetection.test_canny_kernel_thoune_200()
    # TestEdgeDetection.test_canny_kernel_champex_150()
