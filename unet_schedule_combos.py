# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:29:33 2021

@author: zd187
"""


import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/Zach/2D/')

from unet_functions import get_cropped_images
from unet_functions import randomize_split_images
from unet_functions import get_cancer_slices
from unet_functions import dice_coef
from unet_functions import dice_coef_loss
from unet_model import unet
from keras.optimizers import SGD, Adam




# Load cropped images as numpy array
X, y = get_cropped_images()

# Normalize training images and increase non-zero mask values to 1
X = (X - X.min()) / (X.max() - X.min())
y[y>0] = 1
    
# Randomize and split patient ID's according to function random seed
train_ids, val_ids, test_ids = randomize_split_images()

# Create train, validation, and test datasets with only tumor slices
X_train, y_train = get_cancer_slices(train_ids, X, y)
X_val, y_val = get_cancer_slices(val_ids, X, y)
X_test, y_test = get_cancer_slices(test_ids, X, y)

# Add last dimension for U-Net model input
X_train = K.expand_dims(X_train,-1)
X_val = K.expand_dims(X_val,-1)
X_test = K.expand_dims(X_test,-1)
y_train = K.expand_dims(y_train,-1)
y_val = K.expand_dims(y_val,-1)
y_test = K.expand_dims(y_test,-1)




# Plot scheduled combinations
combo_dic = {'SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 2/lr 1e-4/epochs 90': [], 
             'SGD/batch 2/lr 1e-3/epochs 10 -> Adam/batch 2/lr 1e-6/epochs 90': [], 
             'SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 25/lr 1e-3/epochs 90': [],
             'SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 15/lr 1e-3/epochs 90': []}

# First combo
model = unet('HeNormal')
model.compile(optimizer=SGD(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=2, validation_data=(X_val, y_val), epochs=10)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 2/lr 1e-4/epochs 90'].append(score)
    
model.compile(optimizer=SGD(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=2, validation_data=(X_val, y_val), epochs=30)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 2/lr 1e-4/epochs 90'].append(score)
    
# Second combo
model = unet('HeNormal')
model.compile(optimizer=SGD(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=2, validation_data=(X_val, y_val), epochs=10)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> Adam/batch 2/lr 1e-6/epochs 90'].append(score)
    
model.compile(optimizer=Adam(learning_rate=1e-6), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=2, validation_data=(X_val, y_val), epochs=30)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> Adam/batch 2/lr 1e-6/epochs 90'].append(score)

# Third combo
model = unet('HeNormal')
model.compile(optimizer=SGD(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=2, validation_data=(X_val, y_val), epochs=10)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 25/lr 1e-3/epochs 90'].append(score)
    
model.compile(optimizer=SGD(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=25, validation_data=(X_val, y_val), epochs=30)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 25/lr 1e-3/epochs 90'].append(score)

# Fourth combo
model = unet('HeNormal')
model.compile(optimizer=SGD(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=2, validation_data=(X_val, y_val), epochs=10)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 15/lr 1e-3/epochs 90'].append(score)
    
model.compile(optimizer=SGD(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
history = model.fit(X_train, y_train, batch_size=15, validation_data=(X_val, y_val), epochs=30)

for score in history.history['val_dice_coef']:
    combo_dic['SGD/batch 2/lr 1e-3/epochs 10 -> SGD/batch 15/lr 1e-3/epochs 90'].append(score) 
    
    
colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
color_count = 0

for key in combo_dic:   
    epochs = range(1, len(combo_dic[key]) + 1)
    plt.plot(epochs, combo_dic[key], 'b', label=key, color=colors[color_count])
    color_count += 1

plt.rcParams["font.family"] = "Times New Roman"    
plt.title('Validation Dice Score Coefficient', fontdict={'fontsize':20})
plt.xlabel('Epochs', fontdict={'fontsize':16})
plt.ylabel('Dice Score Coefficient', fontdict={'fontsize':16})
plt.legend(loc=4)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('D:/Zach/2D/combo_plots/schedule_combos4.png')
plt.clf()
