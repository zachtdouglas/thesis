# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:57:55 2021

@author: zd187 - Zach Douglas
"""




from keras.layers import Input, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from keras.models import Model




# U-Net model
def unet2d():
    input_img = Input((128, 128, 2))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (input_img)
    b1 = BatchNormalization() (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b1)
    b1 = BatchNormalization() (c1)
    p1 = MaxPooling2D((2, 2)) (b1)
    p1 = Dropout(0.1) (p1)
    
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    b2 = BatchNormalization() (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b2)
    b2 = BatchNormalization() (c2)
    p2 = MaxPooling2D((2, 2)) (b2)
    p2 = Dropout(0.1) (p2)
    
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    b3 = BatchNormalization() (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b3)
    b3 = BatchNormalization() (c3)
    p3 = MaxPooling2D((2, 2)) (b3)
    p3 = Dropout(0.1) (p3)
    
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    b4 = BatchNormalization() (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b4)
    b4 = BatchNormalization() (c4)
    p4 = MaxPooling2D((2, 2)) (b4)
    p4 = Dropout(0.1) (p4)
    
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    b5 = BatchNormalization() (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b5)
    b5 = BatchNormalization() (c5)
    
    u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (b5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    b6 = BatchNormalization() (c6)
    c6 = Dropout(0.1) (b6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b6)
    b6 = BatchNormalization() (c6)
    
    u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (b6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    b7 = BatchNormalization() (c7)
    c7 = Dropout(0.1) (b7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b7)
    b7 = BatchNormalization() (c7)
    
    u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (b7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    b8 = BatchNormalization() (c8)
    c8 = Dropout(0.1) (b8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b8)
    b8 = BatchNormalization() (c8)
    
    u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same') (b8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    b9 = BatchNormalization() (c9)
    c9 = Dropout(0.1) (b9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b9)
    b9 = BatchNormalization() (c9)
    
    output = Conv2D(1, (1, 1), activation='sigmoid') (b9)
    
    model = Model(inputs=[input_img], outputs=[output])
    
    return model