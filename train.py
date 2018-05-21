#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:08:11 2018

@author: rakshith
"""
import math, os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image


PATH = os.getcwd()
# Define data path
DATASET_DIR = PATH + '/Data'

SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.3


if __name__ == "__main__":
    num_dataset_samples = sum([len(files) for r, d, files in os.walk(DATASET_DIR)])

    num_train_steps = math.floor((num_dataset_samples*(1-VALIDATION_SPLIT))/BATCH_SIZE)
    num_valid_steps = math.floor((num_dataset_samples*VALIDATION_SPLIT)/BATCH_SIZE)

    generator = image.ImageDataGenerator(validation_split=VALIDATION_SPLIT)

    batches = generator.flow_from_directory(DATASET_DIR, subset='training', target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = generator.flow_from_directory(DATASET_DIR, subset='validation', target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = keras.applications.resnet50.ResNet50()

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes
    
    filename='model_train_new.csv'
    csv_log=keras.callbacks.CSVLogger(filename, separator=',', append=False)
    
    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
    
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=20, callbacks=[early_stopping, checkpointer, tensorboard_callback, csv_log], validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.save('resnet50_final.h5')
    
    scoreSeg = finetuned_model.evaluate_generator(val_batches, num_dataset_samples*VALIDATION_SPLIT )
    print("Accuracy = ",scoreSeg[1])