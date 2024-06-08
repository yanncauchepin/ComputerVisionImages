#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:56:06 2024

@author: yanncauchepin
"""

import cv2
import pytest
import numpy as np
import modules


# =============================================================================
#   IMAGES
# =============================================================================

path_to_image_thoune= 'images/thoune.jpg'
image_thoune = cv2.imread(path_to_image_thoune)
image_thoune_gray = cv2.imread(path_to_image_thoune, cv2.IMREAD_GRAYSCALE)

path_to_image_anchor_man = 'images/anchor_man.png'
image_anchor_man = cv2.imread(path_to_image_anchor_man)
image_anchor_man_gray = cv2.imread(path_to_image_anchor_man, cv2.IMREAD_GRAYSCALE)

path_to_image_anchor_woman = 'images/anchor_woman.png'
image_anchor_woman = cv2.imread(path_to_image_anchor_woman)
image_anchor_woman_gray = cv2.imread(path_to_image_anchor_woman, cv2.IMREAD_GRAYSCALE)

path_to_image_army = 'images/army.jpg'
image_army = cv2.imread(path_to_image_army)
image_army_gray = cv2.imread(path_to_image_army, cv2.IMREAD_GRAYSCALE)

path_to_image_car = 'images/car.jpg'
image_car = cv2.imread(path_to_image_car)
image_car_gray = cv2.imread(path_to_image_car, cv2.IMREAD_GRAYSCALE)

path_to_image_car_small = 'images/car_small.jpg'
image_car_small = cv2.imread(path_to_image_car_small)
image_car_small_gray = cv2.imread(path_to_image_car_small, cv2.IMREAD_GRAYSCALE)

path_to_image_champex = 'images/champex.jpg'
image_champex = cv2.imread(path_to_image_champex)
image_champex_gray = cv2.imread(path_to_image_champex, cv2.IMREAD_GRAYSCALE)

path_to_image_chessboard = 'images/chessboard.png'
image_chessboard = cv2.imread(path_to_image_chessboard)
image_chessboard_gray = cv2.imread(path_to_image_chessboard, cv2.IMREAD_GRAYSCALE)

path_to_image_gauguin_entre_les_lys = 'images/gauguin_entre_les_lys.jpg'
image_gauguin_entre_les_lys = cv2.imread(path_to_image_gauguin_entre_les_lys)
image_gauguin_entre_les_lys_gray = cv2.imread(path_to_image_gauguin_entre_les_lys, cv2.IMREAD_GRAYSCALE)

path_to_image_gauguin_paintings = 'images/gauguin_paintings.png'
image_gauguin_paintings = cv2.imread(path_to_image_gauguin_paintings)
image_gauguin_paintings_gray = cv2.imread(path_to_image_gauguin_paintings, cv2.IMREAD_GRAYSCALE)

path_to_image_haying = 'images/haying.jpg'
image_haying = cv2.imread(path_to_image_haying)
image_haying_gray = cv2.imread(path_to_image_haying, cv2.IMREAD_GRAYSCALE)

path_to_image_kennedy_space_center = 'images/kennedy_space_center.jpg'
image_kennedy_space_center = cv2.imread(path_to_image_kennedy_space_center)
image_kennedy_space_center_gray = cv2.imread(path_to_image_kennedy_space_center, cv2.IMREAD_GRAYSCALE)

path_to_image_nasa_logo = 'images/nasa_logo.png'
image_nasa_logo = cv2.imread(path_to_image_nasa_logo)
image_nasa_logo_gray = cv2.imread(path_to_image_nasa_logo, cv2.IMREAD_GRAYSCALE)

path_to_image_anchor_query = 'images/anchor_query.png'
image_anchor_query = cv2.imread(path_to_image_anchor_query)
image_anchor_query_gray = cv2.imread(path_to_image_anchor_query, cv2.IMREAD_GRAYSCALE)

# =============================================================================
#   EDGE DETECTION
# =============================================================================

from modules.detection.edge_detection import EdgeDetection
class TestEdgeDetection():

    @staticmethod
    def test_custom_kernel_HPF_thoune_3():
        kernel_3 = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        EdgeDetection.custom_kernel_HPF(kernel=kernel_3, image=image_thoune_gray)
    
    @staticmethod
    def test_custom_kernel_HPF_thoune_5():
        kernel_5 = np.array([[-1, -1, -1, -1, -1],
                              [-1, 1, 2, 1, -1],
                              [-1, 2, 4, 2, -1],
                              [-1, 1, 2, 1, -1],
                              [-1, -1, -1, -1, -1]])
        EdgeDetection.custom_kernel_HPF(kernel=kernel_5, image=image_thoune_gray)
        
    @staticmethod
    def test_gaussian_blur_low_pass_kernel_thoune_17():
        EdgeDetection.gaussian_blur_low_pass_kernel(
            size_kernel=17, image=image_thoune_gray, difference=True)
        
    @staticmethod
    def test_gaussian_blur_low_pass_kernel_champex_17():
        EdgeDetection.gaussian_blur_low_pass_kernel(
            size_kernel=17, image=image_champex_gray, difference=True)

    @staticmethod 
    def test_canny_kernel_thoune_200():
        EdgeDetection.canny_kernel(
            width_kernel=200, height_kernel=200, image=image_thoune_gray)
        
    @staticmethod 
    def test_canny_kernel_champex_150():
        EdgeDetection.canny_kernel(
            width_kernel=150, height_kernel=150, image=image_champex_gray)

# =============================================================================
#   MAIN
# =============================================================================

if __name__=='__main__':
    pass
    # pytest.main()
    
    # TestEdgeDetection.test_custom_kernel_HPF_thoune_3()
    # TestEdgeDetection.test_custom_kernel_HPF_thoune_5()
    # TestEdgeDetection.test_gaussian_blur_low_pass_kernel_thoune_17()
    # TestEdgeDetection.test_gaussian_blur_low_pass_kernel_champex_17()
    # TestEdgeDetection.test_canny_kernel_thoune_200()
    # TestEdgeDetection.test_canny_kernel_champex_150()
