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
from keras.layers import Layer, Lambda, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, InputSpec, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D, UpSampling2D, Multiply, Dropout
import tensorflow as tf
import cv2

def get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis):
    n_kernels = len(sigmas) * len(thetas) * len(lambdas) * len(gammas) * len(psis)
    gabors = []
    for sigma in sigmas:
        for theta in thetas:
            for lambd in lambdas:
                for gamma in gammas:
                    for psi in psis:
                        params = {'ksize': ksize, 'sigma': sigma,
                                  'theta': theta, 'lambd': lambd,
                                  'gamma': gamma, 'psi': psi}
                        #gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)*(1/np.sqrt(3))
                        gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)
                        gf = K.expand_dims(gf, -1)
                        gabors.append(gf)                   
    assert len(gabors) == n_kernels
    initq=K.stack(gabors, axis=-1)
    f_r   = K.zeros(K.int_shape(initq), dtype=None, name=None)
    f_i   = initq
    f_j   = initq
    f_k   = initq
    cat_gabor_4_r = K.concatenate([f_r, -f_i, -f_j, -f_k], axis=-2)
    cat_gabor_4_i = K.concatenate([f_i, f_r, -f_k, f_j], axis=-2)
    cat_gabor_4_j = K.concatenate([f_j, f_k, f_r, -f_i], axis=-2)
    cat_gabor_4_k = K.concatenate([f_k, -f_j, f_i, f_r], axis=-2)
    cat_gabor_4_quaternion = K.concatenate([cat_gabor_4_r, cat_gabor_4_i, cat_gabor_4_j, cat_gabor_4_k], axis=-1)
    return cat_gabor_4_quaternion
    
def convolve_tensor(x, kernel_tensor=None):
    '''
    conv2d
    input tensor: [batch, in_height, in_width, in_channels]
    kernel tensor: [filter_height, filter_width, in_channels, out_channels]
    '''
    # x = tf.image.rgb_to_grayscale(x)
    ksize = (31, 31)  # (31, 31)
    sigmas = [2]
    n_orients = 8
    thetas = np.linspace(0, np.pi, n_orients, endpoint=False)
    lambdas = [8,16,32,64]
    n_phases = 1  # 1, 2, 4
    psis = np.linspace(0, 2*np.pi, n_phases, endpoint=False)
    n_ratios = 2  # 1, 2, 4
    gammas = np.linspace(1, 0, n_ratios, endpoint=False)

    # Generate Gabor filters
    kernel_tensor = get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis)
    
    return K.conv2d(x, kernel_tensor, padding='same')


def gray_get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis):
    n_kernels = len(sigmas) * len(thetas) * len(lambdas) * len(gammas) * len(psis)
    gabors = []
    for sigma in sigmas:
        for theta in thetas:
            for lambd in lambdas:
                for gamma in gammas:
                    for psi in psis:
                        params = {'ksize': ksize, 'sigma': sigma,
                                  'theta': theta, 'lambd': lambd,
                                  'gamma': gamma, 'psi': psi}
                        #gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)*(1/np.sqrt(3))
                        gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)
                        gf = K.expand_dims(gf, -1)
                        gabors.append(gf)                   
    assert len(gabors) == n_kernels
    #print("Created {n_kernels} kernels.")
    initq=K.stack(gabors, axis=-1)
    r=initq
    g=initq
    b=initq
    cat_gabor = K.concatenate([r, g, b], axis=-2)
    cat_gabor_3_rgb = K.concatenate([cat_gabor, cat_gabor, cat_gabor, cat_gabor], axis=-1)
    return cat_gabor_3_rgb
    
def gray_convolve_tensor(x, kernel_tensor=None):
    '''
    conv2d
    input tensor: [batch, in_height, in_width, in_channels]
    kernel tensor: [filter_height, filter_width, in_channels, out_channels]
    '''
    # x = tf.image.rgb_to_grayscale(x)
    ksize = (31, 31)  # (31, 31)
    sigmas = [2,4]
    n_orients = 8
    thetas = np.linspace(0, np.pi, n_orients, endpoint=False)
    lambdas = [8,16,32,64]
    n_phases = 1  # 1, 2, 4
    psis = np.linspace(0, 2*np.pi, n_phases, endpoint=False)
    n_ratios = 1  # 1, 2, 4
    gammas = np.linspace(1, 0, n_ratios, endpoint=False)

    # Generate Gabor filters
    kernel_tensor = gray_get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis)
    
    return K.conv2d(x, kernel_tensor, padding='same')
#
# ConvNet
#
def AQCNN(params):
    
   
    input_seq = Input((96,112,4))
        
        #1
        #trunk branch
    BN2 = QuaternionBatchNormalization( momentum= 0.9, epsilon= 1e-04)(input_seq)
    conv2   = QuaternionConv2D(128, 3, strides=1, padding="same")(BN2)
    act1 = Activation('relu')(conv2)
    conv3   = QuaternionConv2D(128, 1, strides=1, padding="same")(act1)
    act2 = Activation('relu')(conv3)
    conv4   = QuaternionConv2D(128, 3, strides=1, padding="same")(act2)
    act3 = Activation('relu')(conv4)
    conv5   = QuaternionConv2D(64, 1, strides=1, padding="same")(act3)
    act4 = Activation('relu')(conv5)
        #BN3 = QuaternionBatchNormalization( momentum= 0.9, epsilon= 1e-04)(act2)
        
        #soft_mask_branch
    gabor = Lambda(convolve_tensor, arguments={'kernel_tensor': None},  name="gaborlayer")(input_seq)
        #pool3    = MaxPooling2D(2, padding='same')(qgf) #dowm_sampling1
        #skip_connection
    conv6   = QuaternionConv2D(64, 1, strides=1, padding="same")(gabor)
        #direct_connection
        #pool4    = MaxPooling2D(2, padding='same')(conv3) #dowm_sampling2
        #Upsample1 = UpSampling2D([2, 2])(pool4 )#up_sampling1
        #Upsample1 =  Add()([conv6, Upsample1])
        #Upsample2 = UpSampling2D([2, 2])(conv6 )#up_sampling2
        #conv7   = QuaternionConv2D(64, 1, strides=1, padding="same")(pool4)
        #act2 = Activation('relu')(conv7)
    output_soft_mask = Activation('sigmoid')(conv6)
        
    mul = Multiply()([act4, output_soft_mask])
    add1 = Add()([mul, act4])
    pool5   = AveragePooling2D(2, padding='same')(add1)
    flat    = Flatten()(pool5)
    dense  = QuaternionDense(64, activation='relu')(flat)
        #dense2  = QuaternionDense(64, activation='relu')(dense)
        
    #dp = Dropout(0.5)(dense)    
    output = Dense(7, activation='softmax')(dense)
    
    return Model(input_seq, output)  
 


