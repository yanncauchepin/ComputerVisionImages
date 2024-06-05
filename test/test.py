#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:56:06 2024

@author: yanncauchepin
"""

import samples
import samples.classifier_brain_tumor.preprocessing as brain_tumor_preprocessing

if __name__=='__main__':
    
    landscape = brain_tumor_preprocessing.load_dataframe()
    