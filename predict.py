#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:08:11 2018

@author: rakshith
"""

import os, time
import numpy as np
import argparse
from keras.models import load_model
from keras.preprocessing import image
from multiprocessing import Process

PATH = os.getcwd()
CLASSES = ['Cat', 'Dog']
print_string = ''

def wait():
   """This method is used for printing wait process.
   
   """
   animation = "|/-\\"
   idx = 0
   while (1):
       print('[\033[92m INFO \033[0m] '+print_string+' '+animation[idx % len(animation)], end="\r")
       idx += 1
       time.sleep(0.1)
       
def start_animation():
   """Start printing function as another process.
   
   """
   global p1
   p1 = Process(target=wait)
   p1.start()
   
def stop_animation():
   """Kills the process.         
   """
   p1.terminate()

def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default = 'test_images/2.jpeg')
    parser.add_argument('--model', default = 'resnet50_best.h5')
    args = parser.parse_args()
    
    model_path = PATH+'/'+(args.model)
    
    os.system('clear')
    print_string='Loading Model'
    start_animation()
    
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    stop_animation()
    print('[\033[92m INFO \033[0m] Loaded model                      ')
    print('[\033[92m INFO \033[0m] Loaded in: {0:.2f}s'.format(t1-t0))

    test_path = PATH+'/'+(args.image)
    preds = predict(test_path, model)
    y_classes = preds.argmax(axis=-1)
    print('[\033[96m RESULT \033[0m] Probability Vector: ',preds)
    print('[\033[96m RESULT \033[0m] Class: ',CLASSES[y_classes[0]])
