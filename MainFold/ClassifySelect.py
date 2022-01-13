# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:24:21 2021

@author: brian
"""

import os
import shutil
import random

path = "Class/"
percent = 0.9 #train data
os.mkdir("ClassifyTrain")
os.mkdir("ClassifyTrain/train/")
os.mkdir("ClassifyTrain/valid/")
os.mkdir("ClassifyTrain/test/")

for i in os.listdir(path):
    num = len(os.listdir(path + i))
    select_list = random.sample(range(0, num), round(num * percent))
    select_list2 = random.sample(select_list, round(num * 0.9 * 0.8))
    fold = os.listdir(path + "/" + i)
    os.mkdir("ClassifyTrain/train/" + i)
    os.mkdir("ClassifyTrain/valid/" + i)
    os.mkdir("ClassifyTrain/test/" + i)
    
    for data in range(0, num):
        if data in select_list2:        
            shutil.copy("Class/" + i + "/" + fold[data], "ClassifyTrain/train/" + i + "/" + fold[data])
        elif data in select_list: 
            shutil.copy("Class/" + i + "/" + fold[data], "ClassifyTrain/valid/" + i + "/" + fold[data])
        else:  
            shutil.copy("Class/" + i + "/" + fold[data], "ClassifyTrain/test/" + i + "/" + fold[data])
            