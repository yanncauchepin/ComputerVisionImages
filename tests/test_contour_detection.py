#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:17:42 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   CONTOUR DETECTION
# =============================================================================

from ComputerVisionImages.modules.detection.contour_detection import ContourDetection
class TestContourDetection():

    @staticmethod
    def test_contours_thoune():
        ContourDetection.contours(image=image_thoune_gray)
        
    @staticmethod 
    def test_bounding_shape_champex():
        ContourDetection.bounding_shape(image=image_champex)
    
    @staticmethod 
    def test_bounding_shape_gauguin_entre_les_lys():
        ContourDetection.bounding_shape(image=image_gauguin_entre_les_lys)
        
    @staticmethod 
    def test_bounding_shape_nasa_logo():
        ContourDetection.bounding_shape(image=image_nasa_logo)
    
    @staticmethod 
    def test_convex_contours_nasa_logo():
        ContourDetection.convex_contours(image=image_nasa_logo)
    
    @staticmethod 
    def test_convex_contours_champex():
        ContourDetection.convex_contours(image=image_champex)
        
    @staticmethod 
    def test_convex_contours_thoune():
        ContourDetection.convex_contours(image=image_thoune)

# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestContourDetection.test_contours_thoune()
    # TestContourDetection.test_bounding_shape_champex()
    # TestContourDetection.test_bounding_shape_gauguin_entre_les_lys()
    # TestContourDetection.test_bounding_shape_nasa_logo()
    # TestContourDetection.test_convex_contours_nasa_logo()
    # TestContourDetection.test_convex_contours_champex()
    # TestContourDetection.test_convex_contours_thoune()
    