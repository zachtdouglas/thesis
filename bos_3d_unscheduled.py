# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:50:59 2021

@author: zd187
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
import keras.backend as K
import sys
sys.path.append('D:/Zach/')

from unet_functions import get_cropped_images
from unet_functions import get_cropped_pet
from unet_functions import randomize_split_images
from unet_functions import get_ids
from unet_functions import dice_coef
from unet_functions import dice_coef_loss
from unet_functions import get_slices
from unet_functions import get_cancer_slices
from unet3d_model import unet3d
from tensorflow.keras.optimizers import SGD, Adam
from bayes_opt import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold


'''

1 - Load select image data

2 - Match images to train_model

3 - Set optimizer for train_model

4 - Set pbounds

5 - Set number of points

6 - Define optimization history path

7 - Match images to cross val

8 - Set optimizer for cross val

9 - Define cross val history path

10 - Define cross val results path

'''





# Randomize and split patient ID's according to function random seed
train_ids, test_ids, eval_ids = randomize_split_images()

'''Load 3D Images'''
# Load cropped images as numpy array
X_ct3, y3 = get_cropped_images()
X_pt3 = get_cropped_pet()

# Normalize training images and increase non-zero mask values to 1
X_ct3 = (X_ct3 - X_ct3.min()) / (X_ct3.max() - X_ct3.min())
X_pt3 = (X_pt3 - X_pt3.min()) / (X_pt3.max() - X_pt3.min()) 
y3[y3>0] = 1
y3=K.expand_dims(y3,-1)

# Concatenate CT and PET images
X_ct3 = K.expand_dims(X_ct3,-1)
X_pt3 = K.expand_dims(X_pt3,-1)
X3 = np.concatenate((X_ct3, X_pt3), axis=-1)

# Create train and test optimization datasets for fused images
opt_X3_train, opt_y3_train = get_ids(train_ids, X3, y3)
opt_X3_test, opt_y3_test = get_ids(test_ids, X3, y3)

# Create train and test optimization datasets for CT images
opt_Xct3_train, opt_y3_train = get_ids(train_ids, X_ct3, y3)
opt_Xct3_test, opt_y3_test = get_ids(test_ids, X_ct3, y3)

# Create train and test optimization datasets for PET images
opt_Xpt3_train, opt_y3_train = get_ids(train_ids, X_pt3, y3)
opt_Xpt3_test, opt_y3_test = get_ids(test_ids, X_pt3, y3)

#opt_X3_train = opt_X3_train.astype(np.float16())
#opt_y3_train = opt_y3_train.astype(np.float16())
#opt_X3_test = opt_X3_test.astype(np.float16())
#opt_y3_test = opt_y3_test.astype(np.float16())




'''Specify Parameters for Training'''
# Define optimization sets
optX_train = opt_X3_train
optX_test = opt_X3_test
opty_train = opt_y3_train
opty_test = opt_y3_test

# Set parameter bounds for Bayesian optimization
minbatch = 1
maxbatch = 16
minratio = 1
maxratio = 10000

# Specify fused, CT, or PET for cross validation
X_eval = X3

# Specify channel and optimizer for 3D U-Net
channel = 1
opt = SGD

# Set number of points for surrogate model to iterate
exploit = 25
explore = 5

# Set file paths
opt_hist_path = 'D:/Zach/bayes_opt_scheduled_unet3d_test.csv'

cross_val_hist_csvpath = 'D:/Zach/crossval_hist_3dctpet_schedule.csv'
cross_val_hist_xlsxpath = 'D:/Zach/crossval_hist_3dctpet_schedule.xlsx'

cross_val_result_csvpath = 'D:/Zach/crossval_means_3dctpet_schedule.csv'
cross_val_result_xlsxpath = 'D:/Zach/crossval_means_3dctpet_schedule.xlsx'




'''Bayesian Hyperparameter Optimization'''

def train_model(batch1, ratio1):
    lr1 = batch1/ratio1
        
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    model = unet3d(channel)
    
    model.compile(optimizer=opt(learning_rate=lr1), loss=dice_coef_loss, metrics=[dice_coef])
    model.fit(optX_train, opty_train, batch_size=round(batch1), validation_data=(optX_test, opty_test), epochs=50, callbacks=[checkpoint])
    
    scores = model.evaluate(optX_test, optX_test)
    dsc = scores[1] 
    
    return dsc

pbounds = {'batch1': (minbatch, maxbatch), 'ratio1': (minratio, maxratio)}

optimizer = BayesianOptimization(f=train_model, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=explore, n_iter=exploit)

values = optimizer.res

p1 = pd.DataFrame.from_dict(values)
p1.to_csv(opt_hist_path)

batch1 = round(optimizer.max['params']['batch1'])
lr1 = optimizer.max['params']['batch1']/optimizer.max['params']['ratio1']




'''Cross Validation'''

# Cross validation on specified learning schedule
kf = KFold(n_splits=10, shuffle=True, random_state=(1))

metric_dic = {}
best_weight_dic = {}
dsc_counter = 0

for train, test in kf.split(eval_ids):
    X_train, y_train = get_ids(channel, train, X_eval, y3)
    X_test, y_test = get_ids(channel, test, X_eval, y3)
    
    dice_scores = []
    
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    model = unet3d(channel)
    
    start = time()
    model.compile(optimizer=opt(learning_rate=lr1), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    history = model.fit(X_train, y_train, batch_size=batch1, validation_data=(X_test, y_test), epochs=50, callbacks=[checkpoint])
    
    for score in history.history['val_dice_coef']:
        dice_scores.append(score)
        
    elapsed = (time()-start)/60

    metric_dic['DSC ' + str(dsc_counter)] = dice_scores
        
    model.load_weights('best_model.h5')
    
    scores = model.evaluate(X_test, y_test)
    
    best_weight_dic['DSC ' + str(dsc_counter)] = scores[1]   
    best_weight_dic['Precision ' + str(dsc_counter)] = scores[2]
    best_weight_dic['Recall ' + str(dsc_counter)] = scores[3]
    best_weight_dic['Duration ' +str(dsc_counter)] = elapsed

    dsc_counter += 1


p1 = pd.DataFrame.from_dict(metric_dic, orient='index')
p1.to_csv(cross_val_hist_csvpath)
with pd.ExcelWriter(cross_val_hist_xlsxpath) as writer:  
    p1.to_excel(writer, sheet_name='sheet1')
    
p2 = pd.DataFrame.from_dict(best_weight_dic, orient='index')
p2.to_csv(cross_val_result_csvpath)
with pd.ExcelWriter(cross_val_result_xlsxpath) as writer:  
    p2.to_excel(writer, sheet_name='sheet1')  
    
    
    
'''
def bo_model(schedule, model_type, X_train, y_train, X_test, y_test, opt, batch1, batch2, ratio1, ratio2):
    lr1 = batch1/ratio1
    lr2 = batch2/ratio2
    
    delta = 0.001
    
    earlystop1 = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    earlystop2 = EarlyStopping(monitor='val_loss', mode='min', min_delta=delta, patience=10)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    model = model_type
    
    if schedule: 
        model.compile(optimizer=opt(learning_rate=lr1), loss=dice_coef_loss, metrics=[dice_coef])
        model.fit(X_train, y_train, batch_size=round(batch1), validation_data=(X_test, y_test), epochs=50, callbacks=[earlystop1])
        
        model.compile(optimizer=opt(learning_rate=lr2), loss=dice_coef_loss, metrics=[dice_coef])
        model.fit(X_train, y_train, batch_size=round(batch2), validation_data=(X_test, y_test), epochs=50, callbacks=[earlystop2, checkpoint])
        
    else:
        model.compile(optimizer=opt(learning_rate=lr1), loss=dice_coef_loss, metrics=[dice_coef])
        model.fit(X_train, y_train, batch_size=round(batch1), validation_data=(X_test, y_test), epochs=50, callbacks=[checkpoint])
        
    scores = model.evaluate(X_test, y_test)
    
    dsc = scores[1] 
    return dsc
'''





'''2D Image Loading'''
# Load cropped images as numpy array
X_ct2, y2 = get_cropped_images(slices=True)
X_pt2 = get_cropped_pet(slices=True)

# Normalize training images and increase non-zero mask values to 1
X_ct2 = (X_ct2 - X_ct2.min()) / (X_ct2.max() - X_ct2.min())
X_pt2 = (X_pt2 - X_pt2.min()) / (X_pt2.max() - X_pt2.min())
y2[y2>0] = 1
y2 = K.expand_dims(y2,-1)

# Concatenate CT and PET images
X_ct2 = K.expand_dims(X_ct2,-1)
X_pt2 = K.expand_dims(X_pt2,-1)
X2 = np.concatenate((X_ct2, X_pt2), axis=-1)
    
# Create train and test optimizatin datasets for fused images
opt_X2_train, opt_y2_train = get_slices(train_ids, X2, y2)
opt_X2_test, opt_y2_test = get_slices(test_ids, X2, y2)

# Create train and test optimization datasets for CT images
opt_Xct2_train, opt_y2_train = get_slices(train_ids, X_ct2, y2)
opt_Xct2_test, opt_y2_test = get_slices(test_ids, X_ct2, y2)

# Create train and test optimization datasets for PET images
opt_Xpt2_train, opt_y2_train = get_slices(train_ids, X_pt2, y2)
opt_Xpt2_test, opt_y2_test = get_slices(test_ids, X_pt2, y2)




'''2D Image Loading Cancer Slices ONLY'''
# Create train and test optimizatin datasets for fused images with only tumor slices 
opt_X2_train, opt_y2_train = get_cancer_slices(train_ids, X2, y2)
opt_X2_test, opt_y2_test = get_cancer_slices(test_ids, X2, y2)

# Create train and test optimization datasets for CT images with only tumor slices
opt_Xct2_train, opt_y2_train = get_cancer_slices(train_ids, X_ct2, y2)
opt_Xct2_test, opt_y2_test = get_cancer_slices(test_ids, X_ct2, y2)

# Create train and test optimization datasets for PET images with only tumor slices
opt_Xpt2_train, opt_y2_train = get_cancer_slices(train_ids, X_pt2, y2)
opt_Xpt2_test, opt_y2_test = get_cancer_slices(test_ids, X_pt2, y2)




'''3D Image Loading'''
# Load cropped images as numpy array
X_ct3, y3 = get_cropped_images()
X_pt3 = get_cropped_pet()

# Normalize training images and increase non-zero mask values to 1
X_ct3 = (X_ct3 - X_ct3.min()) / (X_ct3.max() - X_ct3.min())
X_pt3 = (X_pt3 - X_pt3.min()) / (X_pt3.max() - X_pt3.min()) 
y3[y3>0] = 1
y3=K.expand_dims(y3,-1)

# Concatenate CT and PET images
X_ct3 = K.expand_dims(X_ct3,-1)
X_pt3 = K.expand_dims(X_pt3,-1)
X3 = np.concatenate((X_ct3, X_pt3), axis=-1)

# Create train and test optimization datasets for fused images
opt_X3_train, opt_y3_train = get_ids(train_ids, X3, y3)
opt_X3_test, opt_y3_test = get_ids(test_ids, X3, y3)

# Create train and test optimization datasets for CT images
opt_Xct3_train, opt_y3_train = get_ids(train_ids, X_ct3, y3)
opt_Xct3_test, opt_y3_test = get_ids(test_ids, X_ct3, y3)

# Create train and test optimization datasets for PET images
opt_Xpt3_train, opt_y3_train = get_ids(train_ids, X_pt3, y3)
opt_Xpt3_test, opt_y3_test = get_ids(test_ids, X_pt3, y3)

#opt_X3_train = opt_X3_train.astype(np.float16())
#opt_y3_train = opt_y3_train.astype(np.float16())
#opt_X3_test = opt_X3_test.astype(np.float16())
#opt_y3_test = opt_y3_test.astype(np.float16())

def select_image_type(image_type):
    if image_type == '3Dfused':
        optX_train = opt_X3_train
        optX_test = opt_X3_test
        opty_train = opt_y3_train
        opty_test = opt_y3_test
        
    elif image_type == '3Dct':
        optX_train = opt_Xct3_train
        optX_test = opt_Xct3_test
        opty_train = opt_y3_train
        opty_test = opt_y3_test
        
    elif image_type == '3Dpt':
        optX_train = opt_Xpt3_train
        optX_test = opt_Xpt3_test
        opty_train = opt_y3_train
        opty_test = opt_y3_test
        
    elif image_type == '2Dfused':
        optX_train = opt_X2_train
        optX_test = opt_X2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == '2Dct':
        optX_train = opt_Xct2_train
        optX_test = opt_Xct2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == '2Dpt':
        optX_train = opt_Xpt2_train
        optX_test = opt_Xpt2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == 'can2Dfused':
        optX_train = opt_X2_train
        optX_test = opt_X2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == 'can2Dct':
        optX_train = opt_Xct2_train
        optX_test = opt_Xct2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == 'can2Dpt':
        optX_train = opt_Xpt2_train
        optX_test = opt_Xpt2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    return optX_train, optX_test, opty_train, opty_test
    
    

pbounds = {'batch1': (1, 16), 'batch2': (1, 16), 'ratio1': (1, 10000), 'ratio2': (1, 10000)}

def bos(schedule, image_type, model_type, opt):
    
    if image_type == '3Dfused':
        optX_train = opt_X3_train
        optX_test = opt_X3_test
        opty_train = opt_y3_train
        opty_test = opt_y3_test
        
    elif image_type == '3Dct':
        optX_train = opt_Xct3_train
        optX_test = opt_Xct3_test
        opty_train = opt_y3_train
        opty_test = opt_y3_test
        
    elif image_type == '3Dpt':
        optX_train = opt_Xpt3_train
        optX_test = opt_Xpt3_test
        opty_train = opt_y3_train
        opty_test = opt_y3_test
        
    elif image_type == '2Dfused':
        optX_train = opt_X2_train
        optX_test = opt_X2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == '2Dct':
        optX_train = opt_Xct2_train
        optX_test = opt_Xct2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == '2Dpt':
        optX_train = opt_Xpt2_train
        optX_test = opt_Xpt2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == 'can2Dfused':
        optX_train = opt_X2_train
        optX_test = opt_X2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == 'can2Dct':
        optX_train = opt_Xct2_train
        optX_test = opt_Xct2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    elif image_type == 'can2Dpt':
        optX_train = opt_Xpt2_train
        optX_test = opt_Xpt2_test
        opty_train = opt_y2_train
        opty_test = opt_y2_test
        
    
    optimizer = BayesianOptimization(f=bo_model(schedule, model_type=model_type, X_train=optX_train, y_train=opty_train, X_test=optX_test, y_test=opty_test, opt=opt), pbounds=pbounds, random_state=1)
    
    optimizer.maximize(init_points=5, n_iter=25)
    
    values = optimizer.res
    
    p1 = pd.DataFrame.from_dict(values)
    
    #p1.to_csv('D:/Zach/bayes_opt_scheduled_unet3d_ctpet_sgd.csv')
    
    lr1 = optimizer.max['params']['batch1']/optimizer.max['params']['ratio1']
    batch1 = round(optimizer.max['params']['batch1'])
    
    if schedule:
        lr2 = optimizer.max['params']['batch2']/optimizer.max['params']['ratio2']
        batch2 = round(optimizer.max['params']['batch2'])

        return lr1, batch1, lr2, batch2, p1
    
    else:
        return lr1, batch1, p1
    
    
    
def cross_val(schedule, model, opt, lr1, bs1, lr2=1, bs2=2):    
    kf = KFold(n_splits=10, shuffle=True, random_state=(1))
    
    metric_dic = {}
    best_weight_dic = {}
    dsc_counter = 0
    
    with tf.device('/gpu:0'):
        for train, test in kf.split(eval_ids):
            X_train, y_train = get_ids(train, X3, y3)
            X_test, y_test = get_ids(test, X3, y3)
            
            X_train = X_train.astype(np.float16())
            X_test = X_test.astype(np.float16())
            y_train = y_train.astype(np.float16())
            y_test = y_test.astype(np.float16())
           
            X_val = X_train[99:]
            X_train = X_train[:99]
            
            y_val = y_train[99:]
            y_train = y_train[:99]
            
            delta = 0.001
            dice_scores = []
            
            earlystop1 = EarlyStopping(monitor='val_loss', mode='min', patience=2)
            earlystop2 = EarlyStopping(monitor='val_loss', mode='min', min_delta=delta, patience=10)
            checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            
            model = model
            start = time()
            
            if schedule:
                model.compile(optimizer=opt(learning_rate=lr1), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                history = model.fit(X_train, y_train, batch_size=bs1, validation_data=(X_val, y_val), epochs=50, callbacks=[earlystop1])
                
                for score in history.history['val_dice_coef']:
                    dice_scores.append(score)
                
                model.compile(optimizer=opt(learning_rate=lr2), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                history = model.fit(X_train, y_train, batch_size=bs2, validation_data=(X_val, y_val), epochs=50, callbacks=[earlystop2, checkpoint])
                
                for score in history.history['val_dice_coef']:
                    dice_scores.append(score)
                    
            else:
                model.compile(optimizer=opt(learning_rate=lr1), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                history = model.fit(X_train, y_train, batch_size=bs1, validation_data=(X_val, y_val), epochs=50, callbacks=[checkpoint])
                
                for score in history.history['val_dice_coef']:
                    dice_scores.append(score)
                
            elapsed = (time()-start)/60
                
            metric_dic['DSC ' + str(dsc_counter)] = dice_scores
                
            model.load_weights('best_model.h5')
            
            scores = model.evaluate(X_test, y_test)
            
            best_weight_dic['DSC ' + str(dsc_counter)] = scores[1]   
            best_weight_dic['Precision ' + str(dsc_counter)] = scores[2]
            best_weight_dic['Recall ' + str(dsc_counter)] = scores[3]
            best_weight_dic['Duration ' +str(dsc_counter)] = elapsed
            
            dsc_counter += 1
            
        return metric_dic, best_weight_dic
    
history, evaluation = cross_val()
    
p1 = pd.DataFrame.from_dict(history, orient='index')
p1.to_csv('D:/Zach/crossval_hist_3dctpet_schedule_sgd.csv')
with pd.ExcelWriter('D:/Zach/crossval_3dctpet_schedule_sgd.xlsx') as writer:  
    p1.to_excel(writer, sheet_name='sheet1')
    
p2 = pd.DataFrame.from_dict(evaluation, orient='index')
p2.to_csv('D:/Zach/crossval_means_3dctpet_schedule_sgd.csv')
with pd.ExcelWriter('D:/Zach/crossval_means_3dctpet_schedule_sgd.xlsx') as writer:  
    p2.to_excel(writer, sheet_name='sheet1')  
    


