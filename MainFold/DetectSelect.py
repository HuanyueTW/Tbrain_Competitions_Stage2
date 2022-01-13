# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 00:15:44 2021

@author: brian
"""

import os 
import shutil
import random

path = "img/"
all_data = os.listdir(path)
num = len(all_data)
os.mkdir("DetectTrain")
os.mkdir("DetectTrain/train")
os.mkdir("DetectTrain/validation")
os.mkdir("DetectTrain/train/images")
os.mkdir("DetectTrain/train/annotations")
os.mkdir("DetectTrain/validation/images")
os.mkdir("DetectTrain/validation/annotations")

percent = 0.85
select_list = random.sample(range(0, num), round(num * (1 - percent)))

for i in range(0, num):
    print(i + 1)
    if i in select_list:        
        shutil.copy("img/" + all_data[i], "DetectTrain/validation/images/" + all_data[i])
        shutil.copy("xml/" + all_data[i][:-4] + ".xml", "DetectTrain/validation/annotations/" + all_data[i][:-4] + ".xml")

    else:        
        shutil.copy("img/" + all_data[i], "DetectTrain/train/images/" + all_data[i])
        shutil.copy("xml/" + all_data[i][:-4] + ".xml", "DetectTrain/train/annotations/" + all_data[i][:-4] + ".xml")
