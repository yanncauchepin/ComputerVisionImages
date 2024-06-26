#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:26:11 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   FEATURES DETECTION
# =============================================================================

from ComputerVisionImages.modules.detection.feature_detection import FeatureDetection
class TestFeatureDetection():
               
    @staticmethod 
    def test_detecting_harris_corner_army():
        FeatureDetection.detecting_harris_corner(image_army_gray)
        
    @staticmethod 
    def test_detecting_harris_corner_chessboard():
        FeatureDetection.detecting_harris_corner(image_chessboard_gray)
        
    @staticmethod 
    def test_detecting_harris_corner_thoune():
        FeatureDetection.detecting_harris_corner(image_thoune_gray)
        
    @staticmethod 
    def test_detecting_harris_corner_champex():
        FeatureDetection.detecting_harris_corner(image_champex_gray)
        
# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestFeatureDetection.test_detecting_harris_corner_army()
    # TestFeatureDetection.test_detecting_harris_corner_chessboard()
    # TestFeatureDetection.test_detecting_harris_corner_thoune()
    # TestFeatureDetection.test_detecting_harris_corner_champex()