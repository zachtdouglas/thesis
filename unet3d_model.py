# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:11:37 2021

@author: zd187
"""


from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv3D, Conv3DTranspose, MaxPool3D, concatenate
from tensorflow.keras.models import Model




def unet3d(channel):
    input_img = Input((96, 96, 96, channel))

    c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (input_img)
    b1 = BatchNormalization() (c1)
    c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b1)
    b1 = BatchNormalization() (c1)
    p1 = MaxPool3D((2, 2, 2)) (b1)
    p1 = Dropout(0.1) (p1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    b2 = BatchNormalization() (c2)
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b2)
    b2 = BatchNormalization() (c2)
    p2 = MaxPool3D((2, 2, 2)) (b2)
    p2 = Dropout(0.1) (p2)
    
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    b3 = BatchNormalization() (c3)
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b3)
    b3 = BatchNormalization() (c3)
    p3 = MaxPool3D((2, 2, 2)) (b3)
    p3 = Dropout(0.1) (p3)
    
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    b4 = BatchNormalization() (c4)
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b4)
    b4 = BatchNormalization() (c4)
    p4 = MaxPool3D((2, 2, 2)) (b4)
    p4 = Dropout(0.1) (p4)
    
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    b5 = BatchNormalization() (c5)
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b5)
    b5 = BatchNormalization() (c5)
    
    u6 = Conv3DTranspose(128, (3, 3, 3), strides=(2, 2, 2), padding='same') (b5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    b6 = BatchNormalization() (c6)
    c6 = Dropout(0.1) (b6)
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b6)
    b6 = BatchNormalization() (c6)
    
    u7 = Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same') (b6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    b7 = BatchNormalization() (c7)
    c7 = Dropout(0.1) (b7)
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b7)
    b7 = BatchNormalization() (c7)
    
    u8 = Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same') (b7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    b8 = BatchNormalization() (c8)
    c8 = Dropout(0.1) (b8)
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b8)
    b8 = BatchNormalization() (c8)
    
    u9 = Conv3DTranspose(16, (3, 3, 3), strides=(2, 2, 2), padding='same') (b8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    b9 = BatchNormalization() (c9)
    c9 = Dropout(0.1) (b9)
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (b9)
    b9 = BatchNormalization() (c9)
    
    output = Conv3D(1, (1, 1, 1), activation='sigmoid') (b9)
    
    model = Model(inputs=[input_img], outputs=[output])
    
    return model
