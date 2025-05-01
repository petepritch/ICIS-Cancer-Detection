# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:19:13 2025

@author: Drew
"""

import random
import torchvision.transforms.functional as F

class DiscreteRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)