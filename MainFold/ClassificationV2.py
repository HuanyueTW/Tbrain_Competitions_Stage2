# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:10:39 2021

@author: brian
"""

from imageai.Detection.Custom import CustomObjectDetection
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import sys
import numpy as np
import os
import shutil
import random
import cv2
import csv

#要改成你所選的模型路徑名稱
ImageAIJsonPath = "DetectTrain/json/detection_config.json"
ImageAIModelPath = "DetectTrain/models/xxx.h5"
InceptionResNetV2Path = 'model-InceptionResNetV2-e102.h5'



def test(cls_list, net, img):
      
    #cor = 0
    #err = 0
    #al = 0
    
        #al += 1
        #ind = i.index('_')
        #name_ans = "1"
    
        
        #img = image.load_img( "ok.jpg", target_size=(100, 100))
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
        if img is None:
            return 0
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = net_final.predict(x)[0]
        top_inds = pred.argsort()[::-1][:3]
        #print(i, cls_list[top_inds[0]])
        '''
        if name_ans == cls_list[top_inds[0]]:
            cor += 1
        else:
            err += 1
            '''
        return cls_list[top_inds[0]]
        
        
#cls_list = os.listdir("D:/場景文字辨識/Clear/done/train")
cls_list = os.listdir("ClassifyTrain/test")
# 影像大小
IMAGE_SIZE = (100, 100)

# 影像類別數
NUM_CLASSES = 979

# 凍結網路層數
FREEZE_LAYERS = 2

start = 3

#類別載入
cls_list = cls_list
#模組載入
for i in range(0,1):
    net = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = net.output
    x = Flatten()(x)

    # 增加 DropOut layer
    x = Dropout(0.5)(x)
    
    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(NUM_CLASSES , activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    
    # 載入舊權重 記得更改到所選的路徑與模型名稱
    net_final.load_weights(InceptionResNetV2Path) 


detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(ImageAIModelPath) 
detector.setJsonPath(ImageAIJsonPath)
detector.loadModel()



for i in os.listdir("target_data/"): 
    
    img = cv2.imread("target_data/" + i)
    detections = detector.detectObjectsFromImage(input_image = "target_data/" + i,
                                             output_image_path = "output.jpg",
                                             minimum_percentage_probability = 50)
    ans = ""
    if (len(detections) > 0):
        
        tar_list = []
        for detection in detections:
            if (detection["name"]=="word"):
                tar_list.append(detection["box_points"]) # xmin ymin xmax ymax
        
        length = len(tar_list)
        if (length > 0):      
        
            tar_list.sort(key = lambda x:x[1])#Y排序
            Yrange = tar_list[length-1][3] - tar_list[0][1]
            tar_list.sort(key = lambda x:x[0])#X排序
            Xrange = tar_list[length-1][2] - tar_list[0][0]
        
            if (Xrange > Yrange):
            
                for n in tar_list:
                    out = img[n[1]:n[3], n[0]:n[2]]
                    #cv2.imwrite("ok" + ".jpg", out)
                    o = test(cls_list, net, out)
                    ans = ans + o
            else:
                tar_list.sort(key = lambda x:x[1])
                for n in tar_list:
                    out = img[n[1]:n[3], n[0]:n[2]]
                    #cv2.imwrite("ok" + ".jpg", out)
                    o = test(cls_list, net, out)
                    ans = ans + o
            
            with open('output.csv', 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(ans)
            #print(ans)
            csvfile.close()
        else:
            with open('output.csv', 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["###"])
            #print("xxx")
            csvfile.close()
    else:
        with open('output.csv', 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["###"])
        #print("xxx")
        csvfile.close()

    