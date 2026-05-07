# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:51:27 2026

@author: TSC
"""

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
for l in os.scandir('./'):
    if l.name.endswith('.png'):
        ar = cv2.imread(l.path, -1)
        plt.figure()
        plt.imshow(ar)
        plt.title(np.max(ar))
