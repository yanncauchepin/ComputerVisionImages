#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:19:13 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   SHAPE DETECTION
# =============================================================================

from ComputerVisionImages.modules.detection.shape_detection import ShapeDetection
class TestShapeDetection():
    
    @staticmethod 
    def test_houghlines_detection_champex():
        ShapeDetection.houghlines_detection(image=image_champex)
        
    @staticmethod 
    def test_houghlines_detection_thoune():
        ShapeDetection.houghlines_detection(image=image_thoune)
        
    @staticmethod 
    def test_houghlines_detection_chessboard():
        ShapeDetection.houghlines_detection(image=image_chessboard)
        
    @staticmethod 
    def test_houghcircles_detection_champex():
        ShapeDetection.houghcircles_detection(image=image_champex)
        
    @staticmethod 
    def test_houghcircles_detection_thoune():
        ShapeDetection.houghcircles_detection(image=image_thoune)
        
    @staticmethod 
    def test_houghcircles_detection_chessboard():
        ShapeDetection.houghcircles_detection(image=image_chessboard)
        

# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestShapeDetection.test_houghlines_detection_champex()
    # TestShapeDetection.test_houghlines_detection_thoune()
    # TestShapeDetection.test_houghlines_detection_chessboard()
    # TestShapeDetection.test_houghcircles_detection_champex()
    # TestShapeDetection.test_houghcircles_detection_thoune()
    # TestShapeDetection.test_houghcircles_detection_chessboard()
    