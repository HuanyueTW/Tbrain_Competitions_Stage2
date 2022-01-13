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
            
            
               
                
                
                
                
                
                
                
                
                
                