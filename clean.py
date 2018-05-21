#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:08:11 2018

@author: rakshith
"""

import os
from os import listdir
from PIL import Image
    
PATH = os.getcwd()
# Define data path
data_path = PATH + '/Data'
data_dir_list = listdir(data_path)

for dataset in data_dir_list:
  print(dataset)
  img_list=listdir(data_path+'/'+dataset)

  print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
  for filename in img_list:

    if filename.endswith('.jpg'):
      try:
        img = Image.open(data_path+'/'+dataset+'/'+filename) # open the image file
        print(img._getexif())
      except (IndexError,AttributeError,OSError) as e:
        os.system('rm '+data_path+'/'+dataset+'/'+filename)
        
    if filename.endswith('.db'):
      os.system('rm '+data_path+'/'+dataset+'/'+filename)
      