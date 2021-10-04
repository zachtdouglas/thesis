# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 18:49:32 2021

@author: zd187
"""


import sys
sys.path.append('D:/Zach/')

from unet_functions import get_cropped_images
from unet_functions import get_cancer_slices
from unet_functions import num_cancer_slices
from unet_functions import display_preds
from unet_functions import randomize_split_images
from unet_functions import dice_coef
from unet_functions import dice_coef_loss
from unet_functions import show_img
from unet_functions import cancer_indices
from unet_model import unet

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD 
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
from unet_functions import get_cropped_pet




'''Use the following process if model weights not saved'''
# Load cropped images as numpy array
X_ct, y = get_cropped_images()
X_pt = get_cropped_pet()

# Normalize training images and increase non-zero mask values to 1
X_ct = (X_ct - X_ct.min()) / (X_ct.max() - X_ct.min())
X_pt = (X_pt - X_pt.min()) / (X_pt.max() - X_pt.min())
y[y>0] = 1
y = K.expand_dims(y,-1)

# Concatenate CT and PET images
X_ct = K.expand_dims(X_ct,-1)
X_pt = K.expand_dims(X_pt,-1)
X = np.concatenate((X_ct, X_pt), axis=-1)

# Randomize and split patient ID's according to function random seed
train_ids, val_ids, test_ids, eval_ids = randomize_split_images()

# Split train and test data for dataset not used in hyperparameter optimization
eval_train = eval_ids[:70]
eval_test = eval_ids[70:]

# Create train, validation, and test datasets with only tumor slices
X_train, y_train = get_cancer_slices(eval_train, X, y)
X_test, y_test = get_cancer_slices(eval_test, X, y)

# Train model
delta = 0.001

earlystop1 = EarlyStopping(monitor='val_loss', mode='min', patience=2)
earlystop2 = EarlyStopping(monitor='val_loss', mode='min', min_delta=delta, patience=10)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model = unet('HeNormal')

model.compile(optimizer=SGD(learning_rate=1.871e-2), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.fit(X_train, y_train, batch_size=10, validation_data=(X_test, y_test), epochs=30, callbacks=[earlystop1])

model.compile(optimizer=SGD(learning_rate=1.409e-2), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.fit(X_train, y_train, batch_size=10, validation_data=(X_test, y_test), epochs=30, callbacks=[earlystop2, checkpoint])

'''Use the following process if model weights have been saved'''
model = unet('HeNormal')
model.load_weights('D:/Zach/best_model.h5')




# Display predictions of model for given patient image slice
total_cancer_slices = num_cancer_slices(78, X, y)
display_preds(model, X, y, 78, 12)


# Display original image or mask   
first_cancer_slice, last_cancer_slice = cancer_indices(78, X, y)
show_img(X_pt, 78, 26)


# Plot train and test dsc values for each patient id
dice_scores = []
for pt in eval_train:
    img, mask = get_cancer_slices([pt], X, y)
    scores = model.evaluate(img, mask)
    dice_scores.append(scores[1])

test_scores = []
for pt in eval_test:
    img, mask = get_cancer_slices([pt], X, y)
    scores = model.evaluate(img, mask)
    test_scores.append(scores[1])

plt.scatter(eval_test, test_scores)
plt.scatter(eval_train, dice_scores)

model.evaluate(X_test, y_test)


# Get dsc for single image
img, mask = get_cancer_slices([78], X, y)
model.evaluate(img, mask)
