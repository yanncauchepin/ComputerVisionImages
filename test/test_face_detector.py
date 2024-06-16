#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:02:55 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np
import modules


# =============================================================================
#   IMAGES
# =============================================================================

path_to_image_army = 'images/army.jpg'
image_army = cv2.imread(path_to_image_army)
image_army_gray = cv2.imread(path_to_image_army, cv2.IMREAD_GRAYSCALE)

# =============================================================================
#   FACE DETECTOR
# =============================================================================

from modules.detector.face_detector import FaceDetector
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