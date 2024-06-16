#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:04:41 2024

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

path_to_image_haying = 'images/haying.jpg'
image_haying = cv2.imread(path_to_image_haying)
image_haying_gray = cv2.imread(path_to_image_haying, cv2.IMREAD_GRAYSCALE)

# =============================================================================
#   PEOPLE DETECTOR
# =============================================================================

from modules.detector.people_detector import PeopleDetector
class TestPeopleDetector():
    
    @staticmethod 
    def test_people_detector_army():
        PeopleDetector.people_detector(image_army)
        
    @staticmethod 
    def test_people_detector_haying():
        PeopleDetector.people_detector(image_haying)
        
# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestPeopleDetector.test_people_detector_army()
    # TestPeopleDetector.test_people_detector_haying()
    