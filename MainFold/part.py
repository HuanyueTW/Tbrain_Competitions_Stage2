# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 08:55:26 2021

@author: brian
"""
import os
import json
import cv2
import matplotlib.pyplot as plt

for i in os.listdir("json"):
    with open("json/" + i, encoding="utf-8") as f:
          
        data = json.load(f)

        #輸出縮排Json
        jsonData_sort = json.dumps(data, sort_keys = True, indent=4)
        #print(jsonData_sort)
        
    
        for j in range(0, len(data['shapes'])): 
            if (data['shapes'][j]['group_id'] == 1 or data['shapes'][j]['group_id'] == 4):#中文字元
            #if (data['shapes'][j]['group_id'] == 0):
                print(data['imagePath'])
                print(data['shapes'][j]['label'], data['shapes'][j]['group_id'])
                print(data['shapes'][j]['points'])
                a = data['shapes'][j]['label']
                a = str(a)
                
                print("---")
                
                x_list, y_list = [], []
                for points in range(0, 4):
                    x_list.append(data['shapes'][j]['points'][points][0])
                    y_list.append(data['shapes'][j]['points'][points][1])         
                xmax, xmin = max(x_list), min(x_list)
                ymax, ymin = max(y_list), min(y_list)
                
                if (ymin < 0):
                    ymin = 0
                if (xmin < 0):
                    xmin = 0
          
                img_ori = cv2.imread("img/" + data['imagePath'])   
                img_after = img_ori[ data['shapes'][j]['points'][0][1] : data['shapes'][j]['points'][2][1], data['shapes'][j]['points'][0][0] : data['shapes'][j]['points'][2][0]]
                #img_after = img_ori[ymin:ymax, xmin:xmax]
                #cv2.imwrite("part_img/" + a + "_" + data['imagePath'][:-4] + "_" + str(j) + ".jpg",img_after)
                cv2.imencode('.jpg', img_after)[1].tofile("part_img/" + a + "_" + data['imagePath'][:-4] + "_" + str(j) + ".jpg")
                #cv2.imencode('.jpg', img_after)[1].tofile("part_plate/" + data['imagePath'][:-4] + "_" + str(j) + ".jpg")
               
                
                del x_list, y_list
        