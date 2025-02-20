# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:43:07 2023

@author: bodyn
"""
#%%
import pandas as pd
import shutil
import os
import natsort 
import numpy as np
from sklearn.model_selection import train_test_split
#%%
label_path = r"D:\참매미의 사투리 및 물천 머신러닝\Total_label_sorted_by_Code_and_Number.txt"
photo_path = r"D:\참매미의 사투리 및 물천 머신러닝\Input\original"
split_path = r"D:\참매미의 사투리 및 물천 머신러닝\Input"

#%%
data = np.array(natsort.natsorted(os.listdir(photo_path),reverse=False))
label_csv = pd.read_csv(label_path)
label = np.array(label_csv['T/F'])
#%%
image_train, image_test, label_train, label_test = train_test_split(data,label,test_size = 0.3, shuffle = True, random_state = 23)

#%%
image_train_true = []; image_train_false = []; image_test_true = []; image_test_false = [];

for f in range(len(image_train)):
    if label_train[f] == 0 :
        image_train_false.append(image_train[f])
    else :
        image_train_true.append(image_train[f])
        
for g in range(len(image_test)) :
    if label_test[g] == 0 :
        image_test_false.append(image_test[g])
    else :
        image_test_true.append(image_test[g])

#%%
os.chdir(photo_path)
train_true = [i for i in os.listdir() if i in image_train_true]
train_false = [i for i in os.listdir() if i in image_train_false]
test_true = [i for i in os.listdir() if i in image_test_true]
test_false = [i for i in os.listdir() if i in image_test_false]
#%%
for i in range(len(train_true)) :
    original_path = photo_path + '\\' + train_true[i]
    save_path = os.path.join(split_path,'Train','T') + '\\' + train_true[i]
    shutil.copy(original_path,save_path)
    
for i in range(len(train_false)) :
    original_path = photo_path + '\\' + train_false[i]
    save_path = os.path.join(split_path,'Train','F') + '\\' + train_false[i]
    shutil.copy(original_path,save_path)
    
for i in range(len(test_true)) :
    original_path = photo_path + '\\' + train_true[i]
    save_path = os.path.join(split_path,'Test','T') + '\\' + train_true[i]
    shutil.copy(original_path,save_path)
    
for i in range(len(test_false)) :
    original_path = photo_path + '\\' + test_false[i]
    save_path = os.path.join(split_path,'Test','F') + '\\' + test_false[i]
    shutil.copy(original_path,save_path)
    

