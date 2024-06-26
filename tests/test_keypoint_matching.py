#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:39:25 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

from ComputerVisionImages.test.data.read_test_data import * 

# =============================================================================
#   KEYPOINT MATCHING
# =============================================================================

from ComputerVisionImages.modules.keypoint.keypoint_matching import KeypointMatching
class TestKeypointMatching():
    
    @staticmethod 
    def test_orb_matching_gauguin():
        KeypointMatching.orb_matching(image_gauguin_entre_les_lys, image_gauguin_paintings)
        
    @staticmethod 
    def test_orb_matching_nasa():
        KeypointMatching.orb_matching(image_nasa_logo, image_kennedy_space_center)
        
    @staticmethod 
    def test_orb_knn_matching_gauguin():
        KeypointMatching.orb_knn_matching(image_gauguin_entre_les_lys, image_gauguin_paintings)
        
    @staticmethod 
    def test_orb_knn_matching_nasa():
        KeypointMatching.orb_knn_matching(image_nasa_logo, image_kennedy_space_center)
        
    @staticmethod 
    def test_orb_knn_ratio_test_matching_gauguin():
        KeypointMatching.orb_knn_ratio_test_matching(image_gauguin_entre_les_lys, image_gauguin_paintings)
        
    @staticmethod 
    def test_orb_knn_ratio_test_matching_nasa():
        KeypointMatching.orb_knn_ratio_test_matching(image_nasa_logo, image_kennedy_space_center)
    
    @staticmethod 
    def test_flann_matching_gauguin():
        KeypointMatching.flann_matching(image_gauguin_entre_les_lys, image_gauguin_paintings)
        
    @staticmethod 
    def test_flann_matching_nasa():
        KeypointMatching.flann_matching(image_nasa_logo, image_kennedy_space_center)
    
    @staticmethod 
    def test_flann_homography_matching_gauguin():
        KeypointMatching.flann_homography_matching(image_gauguin_entre_les_lys, image_gauguin_paintings)
        
    @staticmethod 
    def test_flann_homography_matching_nasa():
        KeypointMatching.flann_homography_matching(image_nasa_logo, image_kennedy_space_center)
    
    @staticmethod 
    def test_scan_for_matches_anchor_tattoo():
        folder_path = 'images/tattoos/'
        KeypointMatching.create_descriptors(folder_path)
        KeypointMatching.scan_for_matches(folder_path, image_anchor_query)
        
        
# =============================================================================
#   MAIN
# =============================================================================

if __name__ == '__main__':
    pass
    # pytest.main()
    
    # TestKeypointMatching.test_orb_matching_gauguin()
    # TestKeypointMatching.test_orb_matching_nasa()
    # TestKeypointMatching.test_orb_knn_matching_gauguin()
    # TestKeypointMatching.test_orb_knn_matching_nasa()
    # TestKeypointMatching.test_orb_knn_ratio_test_matching_gauguin()
    # TestKeypointMatching.test_orb_knn_ratio_test_matching_nasa()
    # TestKeypointMatching.test_flann_matching_gauguin()
    # TestKeypointMatching.test_flann_matching_nasa()
    # TestKeypointMatching.test_flann_homography_matching_gauguin()
    # TestKeypointMatching.test_flann_homography_matching_nasa()
    # TestKeypointMatching.test_scan_for_matches_anchor_tattoo()
    
    