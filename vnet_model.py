# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:20:16 2021

@author: zd187
"""


from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Concatenate, PReLU, Add
from tensorflow.keras.models import Model




def vnet():
    input_img = Input((96, 96, 96, 2))

    # Layer 1
    conv1 = Conv3D(16, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (input_img)
    conv1 = PReLU() (conv1)
    input1 = Concatenate(axis=-1) (8 * [input_img])
    add1 = Add() ([input1, conv1])
    down1 = Conv3D(32, kernel_size=2, strides=2, padding="same", kernel_initializer="he_normal", activation='relu') (add1)
    down1 = PReLU() (down1)

    # Layer 2
    conv2 = Conv3D(32, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (down1)
    conv2 = PReLU() (conv2)
    conv2 = Conv3D(32, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv2)
    conv2 = PReLU() (conv2)
    add2 = Add() ([conv2, down1])
    down2 = Conv3D(64, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add2)
    down2 = PReLU() (down2)

    # Layer 3
    conv3 = Conv3D(64, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (down2)
    conv3 = PReLU() (conv3)
    conv3 = Conv3D(64, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv3)
    conv3 = PReLU() (conv3)
    conv3 = Conv3D(64, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv3)
    conv3 = PReLU() (conv3)
    add3 = Add() ([conv3, down2])
    down3 = Conv3D(128, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add3)
    down3 = PReLU() (down3)

    # Layer 4
    conv4 = Conv3D(128, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (down3)
    conv4 = PReLU() (conv4)
    conv4 = Conv3D(128, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv4)
    conv4 = PReLU() (conv4)
    conv4 = Conv3D(128, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv4)
    conv4 = PReLU() (conv4)
    add4 = Add() ([conv4, down3])
    down4 = Conv3D(256, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add4)
    down4 = PReLU() (down4)

    # Layer 5
    conv5 = Conv3D(256, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (down4)
    conv5 = PReLU() (conv5)
    conv5 = Conv3D(256, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv5)
    conv5 = PReLU() (conv5)
    conv5 = Conv3D(256, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv5)
    conv5 = PReLU() (conv5)
    add5 = Add() ([conv5, down4])
    up5 = Conv3DTranspose(128, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add5)
    up5 = PReLU() (up5)

    # Layer 6
    skipcon6 = Concatenate(axis=4) ([up5, add4])
    conv6 = Conv3D(256, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (skipcon6)
    conv6 = PReLU() (conv6)
    conv6 = Conv3D(256, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv6)
    conv6 = PReLU() (conv6)
    conv6 = Conv3D(256, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv6)
    conv6 = PReLU() (conv6)
    add6 = Add() ([conv6, skipcon6])
    up6 = Conv3DTranspose(64, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add6)
    up6 = PReLU() (up6)

    # Layer 7
    skipcon7 = Concatenate(axis=4) ([up6, add3])
    conv7 = Conv3D(128, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (skipcon7)
    conv7 = PReLU() (conv7)
    conv7 = Conv3D(128, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv7)
    conv7 = PReLU() (conv7)
    conv7 = Conv3D(128, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv7)
    conv7 = PReLU() (conv7)
    add7 = Add() ([conv7, skipcon7])
    up7 = Conv3DTranspose(32, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add7)
    up7 = PReLU() (up7)

    # Layer 8
    skipcon8 = Concatenate(axis=4) ([up7, add2])
    conv8 = Conv3D(64, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (skipcon8)
    conv8 = PReLU() (conv8)
    conv8 = Conv3D(64, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (conv8)
    conv8 = PReLU() (conv8)
    add8 = Add() ([conv8, skipcon8])
    up8 = Conv3DTranspose(16, kernel_size=2, strides=2, kernel_initializer="he_normal", activation='relu') (add8)
    up8 = PReLU() (up8)

    # Layer 9
    skipcon9 = Concatenate(axis=4) ([up8, add1])
    conv9 = Conv3D(32, kernel_size=5, padding="same", kernel_initializer="he_normal", activation='relu') (skipcon9)
    conv9 = PReLU() (conv9)
    add9 = Add() ([conv9, skipcon9])
    conv9 = Conv3D(1, kernel_size=1, padding="same", kernel_initializer="he_normal", activation='relu') (add9)
    conv9 = PReLU() (conv9)

    sigmoid = Conv3D(1, kernel_size=1, padding="same", kernel_initializer="he_normal", activation='sigmoid') (conv9)

    model = Model(inputs=input_img, outputs=sigmoid)
        
    return model