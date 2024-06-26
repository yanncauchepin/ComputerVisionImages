#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:02:55 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   FACE DETECTOR
# =============================================================================

from ComputerVisionImages.modules.detector.face_detector import FaceDetector
class TestFaceDetector():
    
    @staticmethod 
    def test_face_detector_army():
        FaceDetector.face_detector(image_army)
        
# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestFaceDetector.test_face_detector_army()