#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:06:38 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   KEYPOINT DESCRIPTORS
# =============================================================================

from ComputerVisionImages.modules.keypoint.keypoint_descriptor import KeypointDescriptor
class TestKeypointDescriptor():
    
    @staticmethod 
    def test_detecting_sift_keypoint_descriptor_champex():
        KeypointDescriptor.detecting_sift_keypoint_descriptor(image_champex_gray)
        
    @staticmethod 
    def test_detecting_sift_keypoint_descriptor_thoune():
        KeypointDescriptor.detecting_sift_keypoint_descriptor(image_thoune_gray)
        
    @staticmethod 
    def test_detecting_surf_keypoint_descriptor_champex():
        KeypointDescriptor.detecting_surf_keypoint_descriptor(image_champex_gray)
        
    @staticmethod 
    def test_detecting_surf_keypoint_descriptor_thoune():
        KeypointDescriptor.detecting_surf_keypoint_descriptor(image_thoune_gray)
        
# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestKeypointDescriptor.test_detecting_sift_keypoint_descriptor_champex()
    # TestKeypointDescriptor.test_detecting_sift_keypoint_descriptor_thoune()
    # TestKeypointDescriptor.test_detecting_surf_keypoint_descriptor_champex()
    # TestKeypointDescriptor.test_detecting_surf_keypoint_descriptor_thoune()