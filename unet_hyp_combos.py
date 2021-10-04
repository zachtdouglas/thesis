# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:01:37 2021

@author: zd187
"""


import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/Zach/2D/')

from unet_functions import get_cropped_images
from unet_functions import randomize_split_images
from unet_functions import get_cancer_slices
from unet_functions import plot_train_dsc
from unet_functions import plot_train_prec
from unet_functions import plot_train_rec
from unet_functions import plot_val_dsc
from unet_functions import plot_val_prec
from unet_functions import plot_val_rec
from unet_functions import train_model
from keras.optimizers import Adam, SGD, Nadam




# Load cropped images as numpy array
X, y = get_cropped_images()

# Normalize training images and increase non-zero mask values to 1
X = (X - X.min()) / (X.max() - X.min())
y[y>0] = 1
    
# Randomize and split patient ID's according to function random seed
train_ids, val_ids, test_ids, eval_ids = randomize_split_images()

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




# Create combination plots
eval_dic = {}
total_hist = {}

initializers = ['HeNormal','GlorotNormal']
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


# SGD batch size 2
hist_dic, eval_dic, total_hist = train_model(' SGD ', ' SGD2 ', initializers, SGD, learning_rates, 
                                           2, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save SGD (batch size 2) figures
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainSGD_dsc_2.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainSGD_prec_2.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainSGD_rec_2.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValSGD_dsc_2.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValSGD_prec_2.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValSGD_rec_2.png')
plt.clf()

# SGD batch size 10
hist_dic, eval_dic, total_hist = train_model(' SGD ', ' SGD10 ', initializers, SGD, learning_rates, 
                                           10, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)     

# Save SGD (batch size 10) figures                
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainSGD_dsc_10.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainSGD_prec_10.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainSGD_rec_10.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValSGD_dsc_10.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValSGD_prec_10.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValSGD_rec_10.png')
plt.clf()


# SGD batch size 20
hist_dic, eval_dic, total_hist = train_model(' SGD ', ' SGD20 ', initializers, SGD, learning_rates, 
                                           20, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save SGD (batch size 20) figures
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainSGD_dsc_20.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainSGD_prec_20.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainSGD_rec_20.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValSGD_dsc_20.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValSGD_prec_20.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValSGD_rec_20.png')
plt.clf()


# Adam batch size 2
hist_dic, eval_dic, total_hist = train_model(' Adam ', ' Adam2 ', initializers, Adam, learning_rates, 
                                           2, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save Adam (batch size 2) figures                
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainAdam_dsc_2.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainAdam_prec_2.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainAdam_rec_2.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValAdam_dsc_2.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValAdam_prec_2.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValAdam_rec_2.png')
plt.clf()


# Adam batch size 10
hist_dic, eval_dic, total_hist = train_model(' Adam ', ' Adam10 ', initializers, Adam, learning_rates, 
                                           10, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save Adam (batch size 10) figures                               
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainAdam_dsc_10.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainAdam_prec_10.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainAdam_rec_10.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValAdam_dsc_10.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValAdam_prec_10.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValAdam_rec_10.png')
plt.clf()


# Adam batch size 20
hist_dic, eval_dic, total_hist = train_model(' Adam ', ' Adam20 ', initializers, Adam, learning_rates, 
                                           20, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save Adam (batch size 20) figures                                      
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainAdam_dsc_20.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainAdam_prec_20.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainAdam_rec_20.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValAdam_dsc_20.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValAdam_prec_20.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValAdam_rec_20.png')
plt.clf()


# Nadam batch size 2
hist_dic, eval_dic, total_hist = train_model(' Nadam ', ' Nadam2 ', initializers, Nadam, learning_rates, 
                                           2, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save Nadam (batch size 2) figures                                
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainNadam_dsc_2.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainNadam_prec_2.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainNadam_rec_2.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValNadam_dsc_2.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValNadam_prec_2.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValNadam_rec_2.png')
plt.clf()


# Nadam batch size 10
hist_dic, eval_dic, total_hist = train_model(' Nadam ', ' Nadam10 ', initializers, Nadam, learning_rates, 
                                           10, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save Nadam (batch size 10) figures
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainNadam_dsc_10.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainNadam_prec_10.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainNadam_rec_10.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValNadam_dsc_10.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValNadam_prec_10.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValNadam_rec_10.png')
plt.clf()


# Nadam batch size 20
hist_dic, eval_dic, total_hist = train_model(' Nadam ', ' Nadam20 ', initializers, Nadam, learning_rates, 
                                           20, eval_dic, total_hist, X_train, X_val, X_test, y_train, y_val, y_test)

# Save Nadam (batch size 20) figures
plot_train_dsc(hist_dic, 'D:/Zach/2D/TrainNadam_dsc_20.png')
plt.clf()
plot_train_prec(hist_dic, 'D:/Zach/2D/TrainNadam_prec_20.png')
plt.clf()
plot_train_rec(hist_dic, 'D:/Zach/2D/TrainNadam_rec_20.png')
plt.clf()
plot_val_dsc(hist_dic, 'D:/Zach/2D/ValNadam_dsc_20.png')
plt.clf()
plot_val_prec(hist_dic, 'D:/Zach/2D/ValNadam_prec_20.png')
plt.clf()
plot_val_rec(hist_dic, 'D:/Zach/2D/ValNadam_rec_20.png')
plt.clf()




p1 = pd.DataFrame.from_dict(eval_dic, orient='index')
p1.to_csv('D:/Zach/2D/result_data/evaluation.csv')
with pd.ExcelWriter('D:/Zach/2D/result_data/evaluation.xlsx') as writer:  
    p1.to_excel(writer, sheet_name='sheet1')

p2 = pd.DataFrame.from_dict(total_hist, orient='index')
p2.to_csv('D:/Zach/2D/result_data/history.csv')
with pd.ExcelWriter('D:/Zach/2D/result_data/history.xlsx') as writer:  
    p2.to_excel(writer, sheet_name='sheet1')
