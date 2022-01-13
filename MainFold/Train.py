# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:16:17 2021

@author: brian
"""


from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory = "DetectTrain")
trainer.setTrainConfig(object_names_array = ["word", "NumLetter"],
                       batch_size = 4,
                       num_experiments = 60)
trainer.trainModel()