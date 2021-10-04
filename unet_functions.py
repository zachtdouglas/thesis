# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:20:29 2021

@author: zd187 - Zach Douglas
"""

import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
import random
import keras.backend as K
import matplotlib.pyplot as plt
from unet2d_model import unet2d
from time import time
import tensorflow as tf




def get_np_volume_from_sitk(sitk_img):
    trans = (2, 1, 0)
    px_spacing = sitk_img.GetSpacing()
    img_position = sitk_img.GetOrigin()
    np_img = sitk.GetArrayFromImage(sitk_img)
    np_img = np.transpose(np_img, trans)
    return np_img, px_spacing, img_position




def get_cropped_images(slices=False):
    # Create list of patient ID's
    ids_total = []
    ids = []
    dir_path = 'D:/Zach/hecktor2021_train/hecktor_nii/'
    id_path = os.listdir(dir_path)
    
    for _id in id_path:
        if str(_id) != '.DS_Store':
            ids_total.append(_id[:7]) 
            
    for i in range(0, len(ids_total), 3):
        ids.append(ids_total[i])
    
    # Load and crop image slices        
    bb_path = 'D:/Zach/hecktor2021_train/hecktor2021_bbox_training.csv'
    bb_dict = pd.read_csv(bb_path).set_index('PatientID')
    
    if slices:
        X_ct = np.zeros((len(ids)*96, 128, 128))
        y = np.zeros((len(ids)*96, 128, 128))
        
    else:
        X_ct = np.zeros((len(ids), 96, 96, 96))
        y = np.zeros((len(ids), 96, 96, 96))
    
    img_count = 0
    t_slice = 0
    
    for _id in tqdm(ids):
        ct_path = dir_path + _id + '_ct.nii.gz'
        gt_path = dir_path + _id + '_gtvt.nii.gz'
    
        ct_img, spacing, origin = get_np_volume_from_sitk(sitk.ReadImage(ct_path))
        gt_img, spacing, origin = get_np_volume_from_sitk(sitk.ReadImage(gt_path))
        
        bb = np.round((np.asarray([
        bb_dict.loc[_id, 'x1'],
        bb_dict.loc[_id, 'y1'],
        bb_dict.loc[_id, 'z1'],
        bb_dict.loc[_id, 'x2'],
        bb_dict.loc[_id, 'y2'],
        bb_dict.loc[_id, 'z2']
        ]) - np.tile(origin, 2)) / np.tile(spacing, 2)).astype(int) 
        
        ct_img = ct_img[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        gt_img = gt_img[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        
        if slices:
            ct_img = resize(ct_img, (128, 128, 96), anti_aliasing=True)
            gt_img = resize(gt_img, (128, 128, 96), anti_aliasing=True)
            
            for slice_ in range(0, 96):
                    ct_data = ct_img[:, :, slice_]
                    gt_data = gt_img[:, :, slice_]
                    X_ct[t_slice] = ct_data
                    y[t_slice] = gt_data
                    t_slice += 1
                    
        else: 
            ct_img = resize(ct_img, (96, 96, 96), anti_aliasing=True)
            gt_img = resize(gt_img, (96, 96, 96), anti_aliasing=True)
            
            X_ct[img_count, :, :, :] = ct_img
            y[img_count, :, :, :] = gt_img
                
        img_count += 1
        
    return X_ct, y

def get_cropped_pet(slices=False):
    ids_total = []
    ids = []
    dir_path = 'D:/Zach/hecktor2021_train/hecktor_nii/'
    id_path = os.listdir(dir_path)
    
    for _id in id_path:
        if str(_id) != '.DS_Store':
            ids_total.append(_id[:7]) 
            
    for i in range(0, len(ids_total), 3):
        ids.append(ids_total[i])   
    
    bb_path = 'D:/Zach/hecktor2021_train/hecktor2021_bbox_training.csv'
    bb_dict = pd.read_csv(bb_path).set_index('PatientID')
    
    if slices:
        X_pt = np.zeros((len(ids)*96, 128, 128))

    else:
        X_pt = np.zeros((len(ids), 96, 96, 96))

    img_count = 0
    t_slice = 0
        
    for _id in tqdm(ids):
        pt_path = dir_path + _id + '_pt.nii.gz'
        
        pt_img, spacing, origin = get_np_volume_from_sitk(sitk.ReadImage(pt_path))
        
        bb = np.round((np.asarray([
        bb_dict.loc[_id, 'x1'],
        bb_dict.loc[_id, 'y1'],
        bb_dict.loc[_id, 'z1'],
        bb_dict.loc[_id, 'x2'],
        bb_dict.loc[_id, 'y2'],
        bb_dict.loc[_id, 'z2']
        ]) - np.tile(origin, 2)) / np.tile(spacing, 2)).astype(int) 
        
        pt_img = pt_img[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
                
        if slices:
            pt_img = resize(pt_img, (128, 128, 96), anti_aliasing=True)

            for slice_ in range(0, 96):
                    pt_data = pt_img[:, :, slice_]
                    X_pt[t_slice] = pt_data
                    t_slice += 1
                
        else:
            pt_img = resize(pt_img, (96, 96, 96), anti_aliasing=True)
            X_pt[img_count, :, :, :] = pt_img
            
        img_count += 1
    
    return X_pt




# Randomize indices and split into train, validation, and test sets
def randomize_split_images():
    patient_ids = np.arange(224)
    
    rng = np.random.default_rng(18)
    rng.shuffle(patient_ids)
    
    opt_train_ids = patient_ids[:47]
    #opt_val_ids = patient_ids[33:47]
    opt_test_ids = patient_ids[47:67]
    
    eval_ids = patient_ids[67:]
    
    return opt_train_ids, opt_test_ids, eval_ids




# Randomize indices and split into train, validation, and test sets
def randomize_withoutCHMR():
    patient_ids = []
    
    for i in range(0, 55):
        patient_ids.append(i)
        
    for i in range(73, 201):
        patient_ids.append(i)
        
    random.seed(18)    
    random.shuffle(patient_ids)
    
    train_ids = patient_ids[:128]
    val_ids = patient_ids[128:167]
    test_ids = patient_ids[167:183]
    
    return train_ids, val_ids, test_ids




# Obtain 2D slices from patient id list
def get_slices(patient_id_list, X, y):
    index_list = []
    initializer = 0
    
    for pt in patient_id_list:
        start_index = pt * 96
        end_index = start_index + 96
        index_list.append([start_index, end_index])

    for patient_slices in index_list:
        
        for i in tqdm(range(patient_slices[0], patient_slices[1])):
            if initializer == 0:
                y_cancer = y[i:i+1, :, :, :]
                X_cancer = X[i:i+1, :, :, :]
                initializer += 1
            else: 
                y_cancer = np.concatenate((y_cancer, y[i:i+1, :, :, :]))
                X_cancer = np.concatenate((X_cancer, X[i:i+1, :, :, :]))
                    
    return X_cancer, y_cancer




# Pull patient samples from patient id list
def get_ids(channel, id_list, X, y):
    X_sample=np.zeros((len(id_list), 96, 96, 96, channel))
    y_sample=np.zeros((len(id_list), 96, 96, 96, 1))
    
    id_count = 0
    
    for id_ in id_list:
        X_sample[id_count, :, :, :] = X[id_]
        y_sample[id_count, :, :, :] = y[id_]
        
        id_count += 1
        
    return X_sample, y_sample
    



# Obtain cancer slices from patient id list
def get_cancer_slices(patient_id_list, X, y):
    index_list = []
    initializer = 0
    
    for pt in patient_id_list:
        start_index = pt * 96
        end_index = start_index + 96
        index_list.append([start_index, end_index])

    for patient_slices in index_list:
        
        for i in tqdm(range(patient_slices[0], patient_slices[1])):
            if np.sum((y[i])) > 0:
                if initializer == 0:
                    y_cancer = y[i:i+1, :, :, :]
                    X_cancer = X[i:i+1, :, :, :]
                    initializer += 1
                else: 
                    y_cancer = np.concatenate((y_cancer, y[i:i+1, :, :, :]))
                    X_cancer = np.concatenate((X_cancer, X[i:i+1, :, :, :]))
                    
    return X_cancer, y_cancer




# Obtain number of cancer slices for patient id
def num_cancer_slices(patient_id, X, y):
    image, mask = get_cancer_slices([patient_id], X, y)
    return len(mask)




# Return the first cancer slice index from 0 to 96 for given patient id
def find_first_cancer_slice(patient_id, y):
    cancer_range = []
    start = patient_id*96
    stop = start+96
    
    for slices in range(start, stop):
        if np.sum((y[slices]))==0:
            cancer_range.append(0)
        else:
            cancer_range.append(1)
            break
    
    return cancer_range.index(1)




# Return start and end indicies within 0 to 96 for given patient id
def cancer_indices(pt_index, X, y):
    cancer_slices = num_cancer_slices(pt_index, X, y)   
    first_cancer_slice = find_first_cancer_slice(pt_index, y)
    last_cancer_slice = first_cancer_slice+cancer_slices-1
    
    return first_cancer_slice, last_cancer_slice




# Display image or mask by patient id and slice index -- max slice index = 96
def show_img(array, patient_id, slice_index):
    global_index = patient_id*96 + slice_index
    plt.imshow(array[global_index,:,:], cmap='gray')




# Dice score coefficient and loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sqsum_y_true = K.sum(K.square(y_true_f))
    sqsum_y_pred = K.sum(K.square(y_pred_f))
    
    return ((2 * intersection + 1e-5) / (sqsum_y_true + sqsum_y_pred + 1e-5))

def dice_coef_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))




# Visualize model predictions
def display_preds(model, X, y, pt_id, slice_index):
    img_array, img_mask = get_cancer_slices([pt_id], X, y)
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img_array[slice_index,:,:,0], cmap='gray')
    ax[0].contour(img_mask[slice_index,:,:,0], [.5], colors='red')    
    ax[0].set_title('Original Image')

    ax[1].imshow(img_array[slice_index,:,:,0], cmap='gray')
    ax[1].contour(model.predict(img_array[slice_index:slice_index+1,:,:,:])[0,:,:,0], [.5], colors='red')    
    ax[1].set_title('Cancer Predicted')




# Plot training DSC values from a history dictionary
def plot_train_dsc(per_dic, name):
    colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
    color_count = 0
            
    for key in per_dic:
        for metric in per_dic[key]:
            epochs = range(1, len(per_dic[key][metric]) + 1)
            if metric == 'dice_coef':
                plt.plot(epochs, per_dic[key][metric], 'b', label= str(key), color=colors[color_count])
        color_count += 1
        
        plt.rcParams["font.family"] = "Times New Roman"    
        plt.title('Training Dice Score Coefficient', fontdict={'fontsize':20})
        plt.xlabel('Epochs', fontdict={'fontsize':16})
        plt.ylabel('Dice Score Coefficient', fontdict={'fontsize':16})
        plt.legend(loc=4)
        plt.gcf().set_size_inches(18.5, 10.5)

    return plt.savefig(name)




# Plot training precision values from a history dictionary
def plot_train_prec(per_dic, name):
    colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
    color_count = 0
            
    for key in per_dic:
        for metric in per_dic[key]:
            epochs = range(1, len(per_dic[key][metric]) + 1)
            if metric[:4] == 'prec':
                plt.plot(epochs, per_dic[key][metric], 'b', label= str(key), color=colors[color_count])
        color_count += 1
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title('Training Precision', fontdict={'fontsize':20})
    plt.xlabel('Epochs', fontdict={'fontsize':16})
    plt.ylabel('Precision', fontdict={'fontsize':16})
    plt.legend(loc=4)
    plt.gcf().set_size_inches(18.5, 10.5)

    return plt.savefig(name)



# Plot training recall values from a history dictionary
def plot_train_rec(per_dic, name):
    colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
    color_count = 0
            
    for key in per_dic:
        for metric in per_dic[key]:
            epochs = range(1, len(per_dic[key][metric]) + 1)
            if metric[:6] == 'recall':
                plt.plot(epochs, per_dic[key][metric], 'b', label= str(key), color=colors[color_count])
        color_count += 1
        
    plt.rcParams["font.family"] = "Times New Roman"    
    plt.title('Training Recall', fontdict={'fontsize':20})
    plt.xlabel('Epochs', fontdict={'fontsize':16})
    plt.ylabel('Recall', fontdict={'fontsize':16})
    plt.legend(loc=4)
    plt.gcf().set_size_inches(18.5, 10.5)

    return plt.savefig(name)




# Plot validation DSC values from a history dictionary
def plot_val_dsc(per_dic, name):
    colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
    color_count = 0
    
    for key in per_dic:
        for metric in per_dic[key]:
            epochs = range(1, len(per_dic[key][metric]) + 1)
            if metric == 'val_dice_coef':
                plt.plot(epochs, per_dic[key][metric], 'b', label=str(key), color=colors[color_count])
        color_count += 1
        
    plt.rcParams["font.family"] = "Times New Roman"    
    plt.title('Validation Dice Score Coefficient', fontdict={'fontsize':20})
    plt.xlabel('Epochs', fontdict={'fontsize':16})
    plt.ylabel('Dice Score Coefficient', fontdict={'fontsize':16})
    plt.legend(loc=4)
    plt.gcf().set_size_inches(18.5, 10.5)
    
    return plt.savefig(name)




# Plot validation precision values from a history dictionary
def plot_val_prec(per_dic, name):
    colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
    color_count = 0
    
    for key in per_dic:
        for metric in per_dic[key]:
            epochs = range(1, len(per_dic[key][metric]) + 1)
            if metric[:8] == 'val_prec':
                plt.plot(epochs, per_dic[key][metric], 'b', label=str(key), color=colors[color_count])
        color_count += 1
        
    plt.rcParams["font.family"] = "Times New Roman"    
    plt.title('Validation Precision', fontdict={'fontsize':20})
    plt.xlabel('Epochs', fontdict={'fontsize':16})
    plt.ylabel('Precision', fontdict={'fontsize':16})
    plt.legend(loc=4)
    plt.gcf().set_size_inches(18.5, 10.5)
    
    return plt.savefig(name)




# Plot validation recall values from a history dictionary
def plot_val_rec(per_dic, name):
    colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}              
    color_count = 0
    
    for key in per_dic:
        for metric in per_dic[key]:
            epochs = range(1, len(per_dic[key][metric]) + 1)
            if metric[:7] == 'val_rec':
                plt.plot(epochs, per_dic[key][metric], 'b', label=str(key), color=colors[color_count])
        color_count += 1
        
    plt.rcParams["font.family"] = "Times New Roman"    
    plt.title('Validation Recall', fontdict={'fontsize':20})
    plt.xlabel('Epochs', fontdict={'fontsize':16})
    plt.ylabel('Recall', fontdict={'fontsize':16})
    plt.legend(loc=4)
    plt.gcf().set_size_inches(18.5, 10.5)
    
    return plt.savefig(name)




# Train model with specified hyperparameters and store history and evaluation in dictionaries
def train_model(name1, name2, initializers, optimizer_, learning_rates, batch, eval_dictionary, total_history, X_train, X_val, X_test, y_train, y_val, y_test):
    hist_dic = {}
    
    for initializer_ in initializers:
        for lr in learning_rates:
            model = unet2d(initializer_)
            start = time()
            model.compile(optimizer=optimizer_(learning_rate=lr), loss=dice_coef_loss, metrics=[dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            history = model.fit(X_train, y_train, batch_size=batch, validation_data=(X_val, y_val), epochs=100)
            elapsed = (time()-start)/60
            
            per_dic = {}
                    
            hist_dic[str(initializer_) + name1 + str(lr)] = history.history  
            total_history[str(initializer_) + name2 + str(lr)] = history.history
            score = model.evaluate(X_test, y_test, verbose=0, batch_size=32)
                              
            per_dic['DSC'] = score[1]   
            per_dic['Precision'] = score[2]
            per_dic['Recall'] = score[3]       
            per_dic['Duration (min)'] = elapsed 
                
            eval_dictionary[str(initializer_) + name2 + str(lr)] = per_dic                      
                    
    return hist_dic, eval_dictionary, total_history
