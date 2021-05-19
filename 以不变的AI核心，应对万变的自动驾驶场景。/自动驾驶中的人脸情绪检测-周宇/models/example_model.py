#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Parcollet Titouan
from   complexnn                             import *
import keras
from   keras.layers                          import *
from   keras.models                          import Model
import keras.backend                         as     K
import numpy                                 as     np

#
# ConvNet
#

def CNN(params):


    

    if(params.model == 'QCNN'):

        input_seq = Input((96,112,4))
        
        # Conv
        conv    = QuaternionConv2D(32, 3, strides=1, activation='relu', padding="same")(input_seq)
        pool    = AveragePooling2D(2, padding='same')(conv)
        conv2   = QuaternionConv2D(64, 3, strides=1, activation='relu', padding="same")(pool)
        pool2   = AveragePooling2D(4, padding='same')(conv2)

        # Reducing dimension before Dense layer
        flat    = Flatten()(pool2)
        dense  = QuaternionDense(256, activation='relu')(flat)
    
    else:
   
        input_seq = Input((96,112,3))
        # Conv
        conv    = Conv2D(128, 3, strides=1, activation='relu', padding="same")(input_seq)
        pool    = AveragePooling2D(2, padding='same')(conv)
        conv2   = Conv2D(256, 3, strides=1, activation='relu', padding="same")(pool)
        pool2   = AveragePooling2D(4, padding='same')(conv2)

        # Reducing dimension before Dense layer
        flat    = Flatten()(pool2)
        dense  = Dense(1024, activation='relu')(flat)

    output = Dense(7, activation='softmax')(dense)
    return Model(input_seq, output)

#
# DenseNet
#

def DNN(params):

    if(params.model == 'DNN'):
        
        input_seq = Input((96,112,4))
        
        I = Flatten()(input_seq)
        
        h0 = Dense(512, activation='relu')(I)
        d0 = Dropout(0.3)(h0)
        h1 = Dense(512, activation='relu')(d0)
        d1 = Dropout(0.3)(h1)
        h2 = Dense(512, activation='relu')(d1)
    elif(params.model == 'QDNN'):
        
        input_seq = Input((96,112,3))
        
        I = Flatten()(input_seq)
        
        h0 = QuaternionDense(512, activation='relu')(I)
        d0 = Dropout(0.3)(h0)
        h1 = QuaternionDense(512, activation='relu')(h0)
        d1 = Dropout(0.3)(h1)
        h2 = QuaternionDense(512, activation='relu')(h1)

    encoded = Dense(7, activation='softmax')(h2)
    
    return Model(input_seq, encoded);
