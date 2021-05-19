#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Yu Zhou
from   complexnn                             import *
import keras
from   keras.layers                          import *
from   keras.models                          import Model
import keras.backend                         as     K
import numpy                                 as     np
from complexnn.bn import QuaternionBatchNormalization
from keras.layers import Layer, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D, UpSampling2D, Multiply, Dropout



#
# ConvNet
#
def AQCNN(params):
    
   
    input_seq = Input((96,112,4))
        
    BN2 = QuaternionBatchNormalization( momentum= 0.9, epsilon= 1e-04)(input_seq)
    conv2   = QuaternionConv2D(64, 3, strides=1, padding="same")(BN2)
    act1 = Activation('relu')(conv2)
    conv3   = QuaternionConv2D(64, 1, strides=1, padding="same")(act1)
    act2 = Activation('relu')(conv3)
    conv4   = QuaternionConv2D(64, 3, strides=1, padding="same")(act2)
    act3 = Activation('relu')(conv4)
    conv5   = QuaternionConv2D(64, 1, strides=1, padding="same")(act3)
    act4 = Activation('relu')(conv5)
        
        
        
        
        #trunk branch
    pool5   = AveragePooling2D(2, padding='same')(act4)
    flat    = Flatten()(pool5)
    dense  = QuaternionDense(64, activation='relu')(flat)
        
    #dp = Dropout(0.5)(dense)    
    output = Dense(7, activation='softmax')(dense)
    
    return Model(input_seq, output)  
 


