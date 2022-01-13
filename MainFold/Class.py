# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 12:36:40 2021

@author: brian
"""

import os 
import shutil

cla = []
for i in os.listdir("part_img"):
    WordClass = i[0]
    if WordClass not in cla:
        cla.append(WordClass)
        os.mkdir("Class/" + WordClass)
        shutil.copy("part_img/" + i, "Class/" + WordClass + "/" + i)
    else:
        shutil.copy("part_img/" + i, "Class/" + WordClass + "/" + i)
        
#刪調資料小於九筆的類別
for j in os.listdir("Class/"):
    if (len(os.listdir("Class/" + j)) < 9):
        shutil.rmtree("Class/" + j)
    
        