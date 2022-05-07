#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:55:24 2022

@author: mvg
"""
from IPython.display import display
from IPython.display import HTML

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

import copy
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 
import re
import seaborn as sns
import time
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')  

from tensorflow import keras

from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

pd.options.display.max_columns = 300
pd.options.display.max_rows = 300


# Open datasets link for download bellow
# https://www.kaggle.com/competitions/tabular-playground-series-feb-2022/data
train_raw = pd.read_csv("/home/mvg/Documents/Ewps/ML_DL/DL/TPS_022022/data/train.csv")
test_raw = pd.read_csv("/home/mvg/Documents/Ewps/ML_DL/DL/TPS_022022/data/test.csv")
submission_raw = pd.read_csv("/home/mvg/Documents/Ewps/ML_DL/DL/TPS_022022/data/sample_submission.csv")
train_raw.set_index("row_id", inplace=True)
test_raw.set_index("row_id", inplace=True)

# Check nan values
print("The train has {} features with nan values."\
      .format(list(train_raw.isnull().sum().values > 0).count(True)))
print("The test has {} features with nan values."\
      .format(list(test_raw.isnull().sum().values > 0).count(True)))
print("The sample_submission has with  {} features nan values."\
      .format(list(submission_raw.isnull().sum().values > 0).count(True)))

# Exploraratory data analyst
# Check duplicates
train_duplicated_rows = train_raw[train_raw.duplicated()==True].shape[0]
print("\ntrain dataset contains {:,} rows from total {:,} rows.\
Share duplicates = {:.2f}%".format(train_duplicated_rows,
                                  train_raw.shape[0],
                                  100.0*train_duplicated_rows/train_raw.shape[0]))

test_duplicated_rows = test_raw[test_raw.duplicated()==True].shape[0]
print("\ntest dataset contains {:,} rows from total {:,} rows.\
Share duplicates = {:.2f}%".format(test_duplicated_rows,
                                  test_raw.shape[0],
                                  100.0*test_duplicated_rows/test_raw.shape[0]))

# Remove duplicates from train and test datasets
train = train_raw.drop_duplicates() 
test = test_raw.drop_duplicates() 


# Check y range and  dispersion - the uniform distribution
train_target_dispersion = train.target.value_counts(normalize=True)*100
print(train_target_dispersion)
print("\nTotal classes for target :{:}.".format(len(train_target_dispersion.values)))

# Encode target one hot encoder and separate to X and Y
train_x_all = train.iloc[:,:-1].values
train_y_all = train.iloc[:,[-1]].values
test_x_all = test.values
ohe = OneHotEncoder()
ohe.fit(train_y_all)
train_y_all_ohe = ohe.transform(train_y_all).toarray()


# Scaler for array
def npscaler(x_values, scaler="ss"):
    """
    Scale/transform np array. 
    Possible scale/transform option for x features:
    1. None – not scale or trainsform
    2. “ptbc”   Power-transformer by Box-Cox
    3. “ptbc” - .PowerTransformer by Yeo-Johnson’
    4. “rb” - .RobustScaler(
    5. "ss" - StandardScaler    
    For prevent data leakage using separate instance scaler/transformer 
    for each train and test parts.
    Parameters
    ----------
        x_values :np.array with numeric values of features.
        scaler : TYPE - None or str, optional.  The default is None.
    Returns
    -------
        x_vals - scaled/transformed np.array
    """
    scalers = ["ptbc", "ptyj", "rb", "ss"]
    x_vals = np.copy(x_values)
    mms = MinMaxScaler(feature_range=(1, 2))
    ptbc = PowerTransformer(method='box-cox')
    ptyj = PowerTransformer()
    rb = RobustScaler(unit_variance=True)
    ss = StandardScaler()
        
    if scaler == "ptbc":
        x_vals = ptbc.fit_transform(mms.fit_transform(x_vals[:,:]))
                         
    elif scaler == "ptyj":
        x_vals = ptyj.fit_transform(x_vals[:,:])
    
    elif scaler == "rb":
        x_vals = rb.fit_transform(x_vals[:,:]), \
    
    elif scaler == "ss":
        x_vals =  ss.fit_transform(x_vals[:,:])
        
    if scaler not in scalers:
        return "Value error for 'scaler'!Enter \
'ptbc' or", " 'ptyj' or 'rb' or 'ss' value for scaler!"
    return x_vals


# EDA 
#  Find output and input biases 
# Find out put bias vector for y - divide pos==1 to neg==0
output_bias = []
for i in range (train_y_all_ohe.shape[1]):
    neg, pos = np.bincount(train_y_all_ohe[:,0].astype(int))/train_y_all_ohe[:,0].shape[0]
    output_bias.append(np.log(pos/neg))
output_bias = np.array(output_bias)



# Find input bias  for Standard Scaler  using this link
# https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full

train_features = list(train.columns)[:-1]
input_bias=[]
reg_a = (r"(A\d+)")
reg_t = (r"(T\d+)")
reg_g = (r"(G\d+)")
reg_c = (r"(C\d+)")
for name in train_features:
    int_a = int(re.findall(reg_a, name)[0].replace("A",""))
    int_t = int(re.findall(reg_t, name)[0].replace("T",""))
    int_g = int(re.findall(reg_g, name)[0].replace("G",""))
    int_c = int(re.findall(reg_c, name)[0].replace("C",""))
    bias = ((1/4)**10) * math.factorial(10)/(math.factorial(int_a)
                                          * math.factorial(int_t)
                                          * math.factorial(int_g)
                                          * math.factorial(int_c))
    input_bias.append(bias)

# 1. Scale input bias with standard scaler and extract
train_x_all_with_bias = np.append(np.array(input_bias).reshape(1,-1),
                                       train_x_all, axis=0)
train_x_all_with_bias_ss = npscaler(train_x_all_with_bias)
input_bias_ss = train_x_all_with_bias_ss[-1,:]


# Estimate PCA values for dataset
# PCA analysis for train dataset
pca_train=PCA()
pca_train.fit(npscaler(train_x_all))
pca_train_cumsum = np.cumsum(pca_train.explained_variance_ratio_)
pca_train_comp_no = np.array(list(range(1,len(pca_train_cumsum)+1)))
# define number of components with 95% variance
pca_train_comp = np.argmax(pca_train_cumsum >= 0.95) + 1
pca_train_df = pd.DataFrame(data=pca_train_cumsum , 
                           columns=["pca_var_ratio"])

# Check PCA for test dataset
pca_test=PCA()
pca_test.fit(npscaler(test_x_all))
pca_test_cumsum = np.cumsum(pca_test.explained_variance_ratio_)
pca_test_comp_no = np.array(list(range(1,len(pca_test_cumsum)+1)))
# define number of components with 95$ variance
pca_test_comp = np.argmax(pca_test_cumsum >= 0.95) + 1
pca_test_df = pd.DataFrame(data=pca_test_cumsum , 
                           columns=["pca_var_ratio"])

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(pca_test_comp_no, pca_test_cumsum, label="test")
ax.plot(pca_train_comp_no, pca_train_cumsum, label="train")
ax.legend()
# Set major sticks
major_xticks = np.arange(0,len(pca_train_comp_no)+15, 50)
major_yticks = np.arange(0,1.0, 0.1)

# Set minor sticks
minor_xticks = np.arange(0,len(pca_train_comp_no)+15, 5)
minor_yticks = np.arange(0,1.0, 0.025)

# Define major and minor sticks
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

ax.set_yticks(major_yticks);
ax.set_yticks(minor_yticks, minor = True);

ax.grid(visible=True, which="both", axis="both")

# ax labels
ax.set_ylabel('Cummulative sum', fontsize=12, fontweight="bold")
ax.set_xlabel("PCA numbers", fontsize=12, fontweight="bold")
ax.legend()
ax.set_title("Cummulative sum of PCA components for test and train dataset.\
\n Numbers PCA's cumponents  for  95% cumsum for test = {:}, for train = {:}.".format(
    pca_test_comp, pca_train_comp),
             fontsize=13,
             fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()

# Remain 231 PCA components for PCA and transform it
train_pca=PCA(n_components=231, svd_solver='randomized')

train_pca_x_all=train_pca.fit_transform(npscaler(train_x_all))

# Calculate input_bias for pca, convert with SS for futrher data scaling
train_pca_x_all_with_bias = train_pca.fit_transform(train_x_all_with_bias_ss)
train_pca_x_all_with_bias_ss = npscaler(train_pca_x_all_with_bias) 
input_bias_pca = train_pca_x_all_with_bias_ss[-1,:]

# Convert train with PCA
test_pca=PCA(n_components=231, svd_solver='randomized')
test_pca_x_all=test_pca.fit_transform(npscaler(test_x_all))


# Estimate mutual information for classifier
mut_inf_clf = mutual_info_classif(train_x_all, train_y_all)
mut_inf_df = pd.DataFrame(data=list(train.columns)[:-1], columns=['features'])
mut_inf_df["mi_clfs"]=mut_inf_clf
nut_inf_zero_vals = mut_inf_df[mut_inf_df["mi_clfs"]==0].count().values[1] 

# Plot histogram values
fig, ax = plt.subplots(figsize=(12,8))
ax=sns.histplot(data=mut_inf_df, x="mi_clfs", kde=True, ax=ax)
ax.set_title("Histogram of mutual information values. Numbers values with \
zero mutual information values = {}.".format(nut_inf_zero_vals))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Mutual information values", fontsize=12, fontweight="bold")
plt.show()

# Define function for conver and scaling  np.array to tensorflow datasets 


# Convert input and output  biases to to tf constant
tf_output_bias = tf.keras.initializers.Constant(output_bias)
tf_input_bias = tf.keras.initializers.Constant(input_bias_ss)
tf_input_bias_pca = tf.keras.initializers.Constant(input_bias_pca)


def npds_train_test_split(x, y, test_ratio=0.2, batch_sz=512, scaler="ss"):
    """
    Convert, shaffle and scale numpy arrays to tf.data.Dataset.
    Parameters
    ----------
    x : input np.array.
    y : input np.array.
    test_ratio : float, optional, the default is 0.2.
        Ratio for test part to all array  rows. If None return all 
        tf.data.Dataset.
    batch_sz : int, optional. The default is 1024.
        batch size
    scaler : string, optional. The default is "ss".
    Returns
    -------
    tf.data.Dataset.
    """

    if test_ratio != None and test_ratio < 1.0 and isinstance(test_ratio,float):
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=test_ratio, stratify=y,random_state=42)
        x_train, x_test = npscaler(x_train, scaler), npscaler(x_test, scaler)
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_train = ds_train.shuffle(buffer_size=x_train.shape[0])\
            .batch(batch_sz).prefetch(1)
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.shuffle(buffer_size=x_test.shape[0])\
            .batch(batch_sz).prefetch(1)    
        return ds_train, ds_test
    elif test_ratio == None:
        x = npscaler(x)
        ds_all = tf.data.Dataset.from_tensor_slices((x, y))\
            .shuffle(buffer_size=x.shape[0]).batch(batch_sz)
        return ds_all


def clf_adam(shape_x=train_x_all.shape[1], learn_rate=0.005623, 
             drop_out_in=0.35, drop_out_1=0.05, drop_out_2=0.0, drop_out_3=0.1,
             drop_out_4=0.1, input_bias=tf_input_bias, 
             output_bias=tf_output_bias):
    # Define metrics
    metrics_short = [
          tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
    # Create model
    model = Sequential()
    #input layer
    model.add(layers.Dense(units=shape_x, input_shape=(shape_x,),
                           kernel_initializer="GlorotUniform",
                           bias_initializer=input_bias, 
                           activation="relu"))
    model.add(Dropout(rate=drop_out_in))
    model.add(BatchNormalization())
    
    # First hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_1))
    model.add(BatchNormalization())
    
    # Second hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_2))
    model.add(BatchNormalization())
    
    # Trird layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_3))
    model.add(BatchNormalization())
    
    # Fourth layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_4))
    model.add(BatchNormalization())
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learn_rate),
        metrics=metrics_short)
    
    return model


def train_model(model, x_train, y_train_ohe, test_sz=0.2, batch_sz=1024, 
              stop_no=30) :
    callbacks = [EarlyStopping(monitor='categorical_accuracy',mode='max',
                                patience=stop_no,restore_best_weights=True)]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train_ohe,
                                                        test_size=test_sz,
                                                        stratify=y_train_ohe,
                                                        random_state=42)
    
    x_train, x_test = npscaler(x_train), npscaler(x_test)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(buffer_size=x_train.shape[0]).batch(batch_sz)\
        .prefetch(1)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.shuffle(buffer_size=x_test.shape[0]).batch(batch_sz)\
        .prefetch(1)
    ds_start=time.time()
    model_check = copy.copy(model)
    ds_history = model_check.fit(ds_train,
                 epochs=10000,
                 validation_data=ds_test,
                 callbacks=callbacks,
                 verbose=1)
    ds_end=time.time()
    ds_total_time= datetime.timedelta(seconds = (ds_end-ds_start))
    ds_history_df = pd.DataFrame(data=ds_history.history)
    ds_history_df.sort_values(by='val_loss', ascending=True, 
                          inplace=True)
    ds_history_df["epochs"]=ds_history_df.index + 1
    ds_history_df["time"]=ds_total_time
    return model_check, ds_history_df


# Train model without PCA
model_adam, model_adam_hist = train_model(clf_adam(), train_x_all, 
                                          train_y_all_ohe)
# Plot accuracy and loss for all 286 features 
fig, ax = plt.subplots(figsize=(18,12))
ax = sns.lineplot(data=model_adam_hist.set_index("epochs"), ax = ax)

ax.set_title(" Accuracy and Loss for all 286 features.", fontsize=16, 
             fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# Set major sticks
major_xticks = np.arange(0,(divmod(model_adam_hist.shape[0], 50)[0]+1)*50, 50)
major_yticks = np.arange(0,1.1, 0.1)

# Set minor sticks
minor_xticks = np.arange(0,(divmod(model_adam_hist.shape[0], 50)[0]+1)*50, 10)
minor_yticks = np.arange(0,1.1, 0.02)

# Define major and minor sticks
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

ax.set_yticks(major_yticks);
ax.set_yticks(minor_yticks, minor = True);

ax.grid(visible=True, which="both", axis="both")

# ax labels
ax.set_ylabel('Accuracy/Loss', fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
ax.legend()


# evaluate model
ds_xy = npds_train_test_split(train_x_all, train_y_all_ohe, test_ratio=None)
model_adam_acc = model_adam.evaluate(ds_xy)





# Train model with PCA
model_adam_pca, model_adam_hist_pca = train_model(clf_adam(
    shape_x=train_pca_x_all.shape[1], input_bias = tf_input_bias_pca),
                                          train_pca_x_all, 
                                          train_y_all_ohe)

# Plot accuracy and loss for  231 PCA features 
fig, ax = plt.subplots(figsize=(18,12))
ax = sns.lineplot(data=model_adam_hist_pca.set_index("epochs"), ax = ax)

ax.set_title(" Accuracy and Loss for 231 PCA features.", fontsize=16, 
             fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# Set major sticks
major_xticks = np.arange(0,(divmod(model_adam_hist_pca.shape[0], 50)[0]+1)*50, 50)
major_yticks = np.arange(0,1.1, 0.1)

# Set minor sticks
minor_xticks = np.arange(0,(divmod(model_adam_hist_pca.shape[0], 50)[0]+1)*50, 10)
minor_yticks = np.arange(0,1.1, 0.02)

# Define major and minor sticks
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

ax.set_yticks(major_yticks);
ax.set_yticks(minor_yticks, minor = True);

ax.grid(visible=True, which="both", axis="both")

# ax labels
ax.set_ylabel('Accuracy/Loss', fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
ax.legend();

# Evaluate PCA model
ds_pca_xy = npds_train_test_split(train_pca_x_all, train_y_all_ohe, 
                                  test_ratio=None)
model_adam_pca_acc = model_adam_pca.evaluate(ds_pca_xy)


# Predict values
y_predict = model_adam.predict(npscaler(test_x_all))

# convert continuous values to binary using argmax
for row in y_predict:
    row = np.where(row < row[np.argmax(row)],0,1)

y_predict_char = ohe.inverse_transform(y_predict)


# Create pandas dataframe submission
test_drop_columns = list(test.columns)
sample_submission = test.drop(labels=test_drop_columns, axis=1)
sample_submission["target"] = y_predict_char
sample_submission.to_csv("/home/mvg/Documents/Ewps/ML_DL/DL/TPS_022022/data/sample_submission_target.csv")



