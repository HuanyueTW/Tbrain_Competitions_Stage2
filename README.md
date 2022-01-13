# 繁體中文場景文字辨識 程式碼說明

## 組別：這就是我
## 成員：蔣明憲 唐碩謙 黃玥菱 林冠霆 蕭靖騰

## 目錄
- <a href="#dependencies" > 環境套件</a>
- <a href="#installation" > 安裝方式</a>
- <a href="#fold" > 資料夾布局</a>
- <a href="#DetectPretreatment" > 前處理-製作偵測訓練註解檔</a>
- <a href="#ClassificationPretreatment" > 前處理-製作分類訓練樣本</a>
  - <a href="#part" > part.py ： 從 json 裁切出分類訓練樣本</a>
  - <a href="#Class" > Class.py ： 將切出來的樣本按照文字分類到各資料夾</a>
- <a href="#TestPretreatment" > 前處理-製作偵測題目</a>
  - <a href="#Partition" > Partition.py ： 製作public題目資料</a>
  - <a href="#AddBackground" > AddBackground.py ： 把題目資料進行補底</a>
- <a href="#DetectTrainPretreatment" > 前處理-訓練樣本取樣</a>
  - <a href="#DetectTrainSelect" > DetectSelect.py ： 偵測訓練樣本的取樣</a>
  - <a href="#ClassifySelect" > ClassifySelect.py ： 分類訓練樣本的取樣</a>
- <a href="#Training" > 模型訓練</a>
  - <a href="#DetectTrain" > Train.py ： 偵測模型訓練</a>
  - <a href="#ClassificationTrain" > InceptionResNetV2.py ： 分類模型訓練</a>
- <a href="#Main" > 主程式</a>
  - <a href="#DetectClassify" > ClassificationV2.py ： 偵測辨識主程式</a>

<div id="dependencies"></div>

## 環境套件


以下是我們組在運行程式時所使用的環境套件與套件： 
 
 - Python 3.8.8 (Anaconda 2021.05)
 - Tensorflow 2.2.0
 - Keras 2.4.3 
 - ImageAI 2.1.6
 - Opencv-python 4.5.3.56
 - Numpy 1.19.3
 - json 0.9.5
 - shutil 

 如果有加速需求才需要下載這個：

- Tensorflow-gpu 2.2.0

<div id="installation"></div>

## 安裝方式(其他套件亦相同)

**Tensorflow** 或者 **Tensorflow GPU** , 需搭配 CUDA 及 cuDNN 安裝才可使用 GPU 加速
```bash
pip install tensorflow==2.2.0
pip install tensorflow-gpu==2.2.0
```
**ImageAI**
```bash
pip install imageai
```

<div id="fold"></div>

## 資料夾布局(最終)

```
>> MainFold     >> WriteXML.py
                >> part.py
                >> Class.py
                >> Partition.py
                >> AddBackground.py
                >> DetectSelect.py
                >> ClassifySelect.py
                >> Train.py
                >> InceptionResNetV2.py
                >> ClassificationV2.py

                >> img (train 解壓縮)
                >> json (train 解壓縮)
                >> img_public (public 解壓縮)
                >> xml
                >> part_img
                >> Class
                >> AfterData
                >> target_data
                >> DetectTrain
                >> ClassifyTrain
```

## 前處理

<div id="DetectPretreatment"></div>

### 前處理-製作偵測訓練註解檔

### WriteXML.py
#### **將 json 轉換成偵測訓練註解xml檔**
首先準備兩個資料夾一個 txt ，一個是 train 解壓縮後的 json ，另一個是自己新增的資料夾叫做 xml ，檔案只要在跟資料夾同層的位置新增一個 space.txt 即可，接著同一層執行這隻程式，便可以轉換成之後要使用的 xml 註解檔。
這部分的程式內容僅是在修改成我們訓練需要的註解檔。
```python
import os
import shutil
import json
import cv2

json_data = os.listdir('json/')
num = 0
for i in json_data:   
    num += 1
    print(num)
    with open("json/" + i, encoding="utf-8") as f:
        
        data = json.load(f) #讀取json
        
        img = cv2.imread("img/" + data['imagePath'])
        the_width = data['imageWidth']
        the_heigh = data['imageHeight']
        the_depth = img.shape[2]
        
        #XML撰寫準備--------------------------------------
        space = "space.txt"
        foruse = "xml/foruse.txt"
        shutil.copy(space, foruse)
        file = open("xml/foruse.txt", mode = "w")
        
        write=file.write("<annotation>\n<filename>" + data['imagePath'] + "</filename>\n<size>\n")
        write=file.write("<width>" + str(the_width) + "</width>\n<height>" + str(the_heigh) + "</height>\n")
        write=file.write("<depth>" + str(the_depth) + "</depth>\n</size>\n")
        
        for j in range(0, len(data['shapes'])): 
            
            x_list, y_list = [], []
            
            if (data['shapes'][j]['group_id'] == 1):#取出中文字元
                
                for points in range(0, 4):
                    x_list.append(data['shapes'][j]['points'][points][0])
                    y_list.append(data['shapes'][j]['points'][points][1])         
                    xmax, xmin = max(x_list), min(x_list)
                    ymax, ymin = max(y_list), min(y_list)
                
                write=file.write("<object>\n<name>" + "word" + "</name>\n<bndbox>\n")
                write = file.write("<xmin>" + str(xmin) +"</xmin>\n<ymin>" + str(ymin) + "</ymin>\n")        
                write = file.write("<xmax>" + str(xmax) +"</xmax>\n<ymax>" + str(ymax) + "</ymax>\n</bndbox>\n</object>\n")
            
            elif (data['shapes'][j]['group_id'] == 2):#取出英文數字字串
                
                for points in range(0, 4):
                    x_list.append(data['shapes'][j]['points'][points][0])
                    y_list.append(data['shapes'][j]['points'][points][1])         
                    xmax, xmin = max(x_list), min(x_list)
                    ymax, ymin = max(y_list), min(y_list)
                
                write = file.write("<object>\n<name>" + "NumLetter" + "</name>\n<bndbox>\n")
                write = file.write("<xmin>" + str(xmin) +"</xmin>\n<ymin>" + str(ymin) + "</ymin>\n")        
                write = file.write("<xmax>" + str(xmax) +"</xmax>\n<ymax>" + str(ymax) + "</ymax>\n</bndbox>\n</object>\n")
            del x_list, y_list
            
        write = file.write("</annotation>")    
        file.close()
        os.rename("xml/foruse.txt", "xml/" + data['imagePath'][:-4] + ".xml")
```

<div id="ClassificationPretreatment"></div>

### 前處理-製作分類訓練樣本

<div id="part"></div>

### part.py


#### **從 json 裁切出分類訓練樣本**
使用 train 解壓縮後的 json 跟 img 資料夾，並再同層建立一個資料夾叫做 part_img 即可，將程式放在同曾位置執行，便會將每個繁體字從原照片切出。
```python

import os
import json
import cv2
import matplotlib.pyplot as plt

for i in os.listdir("json"):
    with open("json/" + i, encoding="utf-8") as f:
          
        data = json.load(f)

        #輸出縮排Json
        jsonData_sort = json.dumps(data, sort_keys = True, indent=4)  
    
        for j in range(0, len(data['shapes'])): 
            if (data['shapes'][j]['group_id'] == 1 or data['shapes'][j]['group_id'] == 4):#中文單字
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

                cv2.imencode('.jpg', img_after)[1].tofile("part_img/" + a + "_" + data['imagePath'][:-4] + "_" + str(j) + ".jpg")
                
                del x_list, y_list
```

<div id="Class"></div>

### Class.py


#### **將切出來的樣本按照文字分類到各資料夾**
與前步驟製作好的 part_img 資料夾，於同層新增 Class 資料夾並執行該程式，便可以將資料按照類別放置完成，會在 Class 資料夾裡頭出現2537個類別資料夾。
```python
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
    
```

<div id="TestPretreatment"></div>

### 前處理-製作偵測題目

<div id="Partition"></div>

### Partition.py


#### **製作public題目資料**
將官方 public 解壓縮後會的到資料夾 img_public 跟一個 csv 檔案，我們需要在同層建立新資料夾叫做 AfterData 並執行該程式碼，就會將原本兩千張的原圖切成10643張題目照片，並存於 AfterData 中。
```python
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
```

<div id="AddBackground"></div>

### AddBackground.py


#### **把題目資料進行補底**
首先將程式一樣放置在跟剛剛做出的AfterData同一層，並且準備一張圖片是1024*1024的白底照片並命名為 space1024.jpg，以及同一層再新增資料夾叫做target_data，再執行程式便可以生成補底過的題目。
```python
import os
import numpy as np
import cv2

for i in os.listdir("part_data"):
    img = cv2.imread("part_data/" + i)
    space = cv2.imread("space1024.jpg")
    y = img.shape[0]
    x = img.shape[1]

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
     
    elif(img.shape[0] < 1025 and img.shape[1] < 1025):
        
        back = space
        back[512 - round(y/2) : 512 - round(y/2) + y, 512 - round(x/2) : 512 - round(x/2) + x] = img
    
    cv2.imwrite("target_data/" + i, back)
```

### 前處理-訓練樣本取樣
<div id="DetectTrainPretreatment"></div>

### DetectSelect.py
<div id="DetectTrainSelect"></div>

將這支程式與 train 解壓縮後的 img 及 我們前面做好的 xml 放同一層執行，便可以將切好的資料按照85%、15%的比例並且放置成 ImageAI 訓練所要得資料結構，這支程式執行結果會出現新資料夾叫做 DetectTrain，之後訓練跟偵測任務都會用到。
```python
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

```

### ClassifySelect.py
<div id="ClassifySelect"></div>

將這支程式與前面製作好的 Class 資料夾放同層之後執行，便會自動生成一個資料夾叫做 ClassifyTrain，裡面會出現三個按照比例分配好的 Train、Valid、Test 資料夾。
```python
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
```


<div id="Training"></div>

## 模型訓練

<div id="DetectTrain"></div>

### 偵測模型訓練
### Train.py
### **注意這裡直接運行會有問題，需要先把ImageAI套件中，路徑imageai/Deetection/Custom/_init_.py的第21行註解掉**
### #tf.config.run_functions_eagerly(True)
只要放置在已經處裡好的訓練資料夾(按比例及資料夾結構)同一層，便可以直接訓練，不需要改動任何模型或訓練參數， batch size 可看GPU的記憶體來做調整， num_exoeriments 大概設置五十上下即可，訓練時每一次只要 loss 更低就會儲存一次，因此我們只需要關注訓練資料夾內的 models 跟 json 裡的檔案即可，這兩個檔案會在要做偵測的時候用到。
```python
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory = "DetectTrain") #指定到訓練資料夾
trainer.setTrainConfig(object_names_array = ["word", "NumLetter"],
                       batch_size = 4,
                       num_experiments = 60)
trainer.trainModel()
```


### 分類模型訓練
<div id="ClassificationTrain"></div>

### InceptionResNetV2.py
一樣把這支程式跟做好的分類訓練資料放置在同一層，並指定資料夾位置(ClassifyTrain)，裡面會出現三個按照比例分配好的)即可，所有參數都是遵照我們比賽的實際環境，但要注意一點是 NUM_CLASSES 如果只是按照前步驟做出來的資料夾可能沒有這麼多，因此請彈性調整成手邊有的資料類別數量；另外還有要注意的就是儲存權重檔案的路徑，也請記得換成符合自己需要。其餘部分的調整可以參考程式碼內註解。
```python
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 資料路徑
DATASET_PATH  = 'ClassifyTrain'

# 影像大小
IMAGE_SIZE = (100, 100)

# 影像類別數
NUM_CLASSES = 979

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 16

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 2

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-resnet50-e150--No-zoom_range.h5'

# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2, #水平移
                                   height_shift_range=0.2, #垂直移
                                   shear_range=0.2, #x,y一個固定平移
                                   #zoom_range=0.2, #縮放
                                   channel_shift_range=10, #變色濾鏡
                                   #horizontal_flip=True, 垂直翻轉
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# 以訓練好的 InceptionResNetV2 為基礎來建立模型，
# 捨棄 InceptionResNetV2 頂層的 fully connected layers
net = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
for i in range(1,101):
    WEIGHTS_FINAL = "model-InceptionResnetV2-e" + str(i * 2)
    net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)
    if i >= 1:
        # 儲存訓練好的模型 路徑請自行調整
        net_final.save(WEIGHTS_FINAL + ".h5")
```

<div id="Main"></div>

## 預測主程式
<div id="DetectClassify"></div>

### ClassificationV2.py
首先準備好四樣東西，一個是要被偵測的題目資料，兩個是剛剛訓練出來的偵測權重( models 內的 .h5 檔案 )及 Anchor box 設定檔( json 內的 .json 檔案)，最後則是分類模型訓練好後的權重檔，將這些檔案連同程式放在同層位置，便可以直接執行最後的主程式，結果會得到一個 output.csv。
```python
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

        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
        if img is None:
            return 0
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = net_final.predict(x)[0]
        top_inds = pred.argsort()[::-1][:3]

        return cls_list[top_inds[0]]
        
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

                    o = test(cls_list, net, out)
                    ans = ans + o
            else:
                tar_list.sort(key = lambda x:x[1])
                for n in tar_list:
                    out = img[n[1]:n[3], n[0]:n[2]]

                    o = test(cls_list, net, out)
                    ans = ans + o
            
            with open('output.csv', 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(ans)

            csvfile.close()
        else:
            with open('output.csv', 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["###"])

            csvfile.close()
    else:
        with open('output.csv', 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["###"])

        csvfile.close()
```
