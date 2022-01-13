# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:06:10 2021

@author: brian
"""

import os
import numpy as np
import cv2


for i in os.listdir("part_data"):
    img = cv2.imread("part_data/" + i)
    space = cv2.imread("space1024.jpg")
    y = img.shape[0]
    x = img.shape[1]

    #print(img.shape) # y, x, depth
    if(img.shape[0] > 2048 or img.shape[1] > 2048):
        
        back = cv2.resize(space, (3096, 3096))
        back[1548 - round(y/2) : 1548 - round(y/2) + y, 1548 - round(x/2) : 1548 - round(x/2) + x] = img
     
    elif(img.shape[0] > 1024 or img.shape[1] > 1024):
        
        back = cv2.resize(space, (2048, 2048))
        back[1024 - round(y/2) : 1024 - round(y/2) + y, 1024 - round(x/2) : 1024 - round(x/2) + x] = img
        
    elif(img.shape[0] <= 50 and img.shape[1] <= 50):
             
        img = cv2.resize(img, (x*4, y*4), interpolation=cv2.INTER_CUBIC)
        x = x*4
        y = y*4
        back = cv2.resize(space, (512, 512))
        back[256 - round(y/2) : 256 - round(y/2) + y, 256 - round(x/2) : 256 - round(x/2) + x] = img
    
    elif(img.shape[0] <= 50 or img.shape[1] <= 50):
        
        le = x
        if (y > x):
            le = y
        if (le % 2 != 0):
            le = le + 101
        else:
            le = le + 100
        
        
        img = cv2.resize(img, (x*2, y*2), interpolation=cv2.INTER_CUBIC)
        x = x*2
        y = y*2
        back = cv2.resize(space, (le * 2, le * 2))
        back[le - round(y/2) : le - round(y/2) + y, le - round(x/2) : le - round(x/2) + x] = img
    
    #elif(img.shape[0] < 256 and img.shape[1] < 256):
        
        #img = cv2.resize(img, (x*2, y*2), interpolation=cv2.INTER_CUBIC)
        #x = x*2
        #y = y*2
        #back = cv2.resize(space, (512, 512))
        #back[256 - round(y/2) : 256 - round(y/2) + y, 256 - round(x/2) : 256 - round(x/2) + x] = img
    
    #elif(img.shape[0] < 512 and img.shape[1] < 512):
        
    #    back = cv2.resize(space, (512, 512))
    #    back[256 - round(y/2) : 256 - round(y/2) + y, 256 - round(x/2) : 256 - round(x/2) + x] = img
         
    elif(img.shape[0] < 1025 and img.shape[1] < 1025):
        
        back = space
        back[512 - round(y/2) : 512 - round(y/2) + y, 512 - round(x/2) : 512 - round(x/2) + x] = img
    
    cv2.imwrite("target_data/" + i, back)
    
        
    
    
