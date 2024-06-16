#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:39:25 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np

import sys
sys.path.append('.')
import modules


# =============================================================================
#   IMAGES
# =============================================================================

path_to_image_thoune= 'test/images/thoune.jpg'
image_thoune = cv2.imread(path_to_image_thoune)
image_thoune_gray = cv2.imread(path_to_image_thoune, cv2.IMREAD_GRAYSCALE)

path_to_image_anchor_man = 'test/images/anchor_man.png'
image_anchor_man = cv2.imread(path_to_image_anchor_man)
image_anchor_man_gray = cv2.imread(path_to_image_anchor_man, cv2.IMREAD_GRAYSCALE)

path_to_image_anchor_woman = 'test/images/anchor_woman.png'
image_anchor_woman = cv2.imread(path_to_image_anchor_woman)
image_anchor_woman_gray = cv2.imread(path_to_image_anchor_woman, cv2.IMREAD_GRAYSCALE)

path_to_image_army = 'test/images/army.jpg'
image_army = cv2.imread(path_to_image_army)
image_army_gray = cv2.imread(path_to_image_army, cv2.IMREAD_GRAYSCALE)

path_to_image_car = 'test/images/car.jpg'
image_car = cv2.imread(path_to_image_car)
image_car_gray = cv2.imread(path_to_image_car, cv2.IMREAD_GRAYSCALE)

path_to_image_car_small = 'test/images/car_small.jpg'
image_car_small = cv2.imread(path_to_image_car_small)
image_car_small_gray = cv2.imread(path_to_image_car_small, cv2.IMREAD_GRAYSCALE)

path_to_image_champex = 'test/images/champex.jpg'
image_champex = cv2.imread(path_to_image_champex)
image_champex_gray = cv2.imread(path_to_image_champex, cv2.IMREAD_GRAYSCALE)

path_to_image_chessboard = 'test/images/chessboard.png'
image_chessboard = cv2.imread(path_to_image_chessboard)
image_chessboard_gray = cv2.imread(path_to_image_chessboard, cv2.IMREAD_GRAYSCALE)

path_to_image_gauguin_entre_les_lys = 'test/images/gauguin_entre_les_lys.jpg'
image_gauguin_entre_les_lys = cv2.imread(path_to_image_gauguin_entre_les_lys)
image_gauguin_entre_les_lys_gray = cv2.imread(path_to_image_gauguin_entre_les_lys, cv2.IMREAD_GRAYSCALE)

path_to_image_gauguin_paintings = 'test/images/gauguin_paintings.png'
image_gauguin_paintings = cv2.imread(path_to_image_gauguin_paintings)
image_gauguin_paintings_gray = cv2.imread(path_to_image_gauguin_paintings, cv2.IMREAD_GRAYSCALE)

path_to_image_haying = 'test/images/haying.jpg'
image_haying = cv2.imread(path_to_image_haying)
image_haying_gray = cv2.imread(path_to_image_haying, cv2.IMREAD_GRAYSCALE)

path_to_image_kennedy_space_center = 'test/images/kennedy_space_center.jpg'
image_kennedy_space_center = cv2.imread(path_to_image_kennedy_space_center)
image_kennedy_space_center_gray = cv2.imread(path_to_image_kennedy_space_center, cv2.IMREAD_GRAYSCALE)

path_to_image_nasa_logo = 'test/images/nasa_logo.png'
image_nasa_logo = cv2.imread(path_to_image_nasa_logo)
image_nasa_logo_gray = cv2.imread(path_to_image_nasa_logo, cv2.IMREAD_GRAYSCALE)

path_to_image_anchor_query = 'test/images/anchor_query.png'
image_anchor_query = cv2.imread(path_to_image_anchor_query)
image_anchor_query_gray = cv2.imread(path_to_image_anchor_query, cv2.IMREAD_GRAYSCALE)


# =============================================================================
#   KEYPOINT MATCHING
# =============================================================================

from modules.keypoint.keypoint_matching import KeypointMatching
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
    
    