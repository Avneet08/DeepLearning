# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:40:19 2018

@author: Avneet
"""
#Part 1-building CNN
from keras.models import Sequential # to initialize a nn as sequence of layers
from keras.layers import Dense # to fully connect the whole cnn
from keras.layers import Flatten # flatten the layer
from keras.layers import Convolution2D # Convuloution step to deal with images
from keras.layers import MaxPooling2D # pooling stepp

#Initialize the cnn
classifier=Sequential()

#add diifernt layers
#
##
###
# first layer is Convolutional layer which is composed of various feature map
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

# max pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
#flatening layer
classifier.add(Flatten())
# now we have to add a classic nn
# its a fully connected layer\
# we are adding a hidden layer
classifier.add(Dense(units=128,activation='relu'))
#output layer
classifier.add(Dense(units=1,activation='sigmoid'))

#compiling the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# part2 fitting into the cnn
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=200)