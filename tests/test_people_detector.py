#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:04:41 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   PEOPLE DETECTOR
# =============================================================================

from ComputerVisionImages.modules.detector.people_detector import PeopleDetector
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
    