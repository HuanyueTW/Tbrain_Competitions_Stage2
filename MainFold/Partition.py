# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:49:05 2021

@author: brian
"""

import os
import cv2
import csv

print(os.listdir("AfterData"))

# 開啟 CSV 檔案
with open('Task2_Public_String_Coordinate.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)
  num = 0
  # 以迴圈輸出每一列
  for row in rows:
    num+=1
    x_list, y_list = [], []
    img_ori = cv2.imread("img_public/" + row[0] + ".jpg") 
    print(row[0])
    
    for i in range(1,len(row)):
        if (i%2==1):
            x_list.append(int(row[i]))
        else:
            y_list.append(int(row[i]))  

    xmax, xmin = max(x_list), min(x_list)
    ymax, ymin = max(y_list), min(y_list) 
    
    print(num, xmin, xmax, ymin, ymax)
    print(img_ori.shape)
    img_after = img_ori[int(ymin) : int(ymax), int(xmin) : int(xmax)]
    cv2.imwrite("AfterData/" + row[0] + "_" + str(num).zfill(4) + ".jpg", img_after)
    del x_list, y_list
    