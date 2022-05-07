#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:09:32 2022

@author: mvg
"""
# Ray tune modules
import argparse
import os

from filelock import FileLock
from tensorflow.keras.datasets import mnist

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback

# Other modules
from IPython.display import display
from IPython.display import HTML

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

import copy
import datetime
import keras_tuner as kt
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import re
import time
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')  

from matplotlib import ticker

from tensorflow import keras
from keras.regularizers import l1
from keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras import initializers
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import max_norm

# ADAM weight decay
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from torch_optimizer import QHAdam
from tensorflow_addons.optimizers import AdamW # WeightDecayScheduler
from keras import backend as K
import os
import sys



pd.options.display.max_columns = 300
pd.options.display.max_rows = 300

# Open datasets
# Dataset link -  https://www.kaggle.com/c/tabular-playground-series-feb-2022

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
print("\nThe train dataset contains {:,} duplicate rows from total {:,} rows.\
Share duplicates = {:.2f}%".format(train_duplicated_rows,
                                  train_raw.shape[0],
                                  100.0*train_duplicated_rows/train_raw.shape[0]))

test_duplicated_rows = test_raw[test_raw.duplicated()==True].shape[0]
print("\nthe test dataset contains {:,} duplicate rows from total {:,} rows.\
Share duplicates = {:.2f}%".format(test_duplicated_rows,
                                  test_raw.shape[0],
                                  100.0*test_duplicated_rows/test_raw.shape[0]))

# Remove duplicates from train and test datasets
train = train_raw.drop_duplicates() 
test = test_raw.drop_duplicates() 

# Scaler for array
def npscaler(x_values, scaler="ss"):
    """
    Scale/transform np array. 
    Possible scale/transform option for x features:
    1. None – not scale or trainsform
    2. “ptbc”   Power-transformer by Box-Cox
    3. “ptbc” - .PowerTransformer by Yeo-Johnson’
    4. “rb” - .RobustScaler
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
    scalers = ["ptbc", "ptyj", "rb", "ss", "ss-mms"]
    x_vals = np.copy(x_values)
    mms = MinMaxScaler(feature_range=(-1, 1))
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
    elif scaler == "ss-mms":
        x_vals =  mms.fit_transform(ss.fit_transform(x_vals[:,:]))
        
    if scaler not in scalers:
        return "Value error for 'scaler'!Enter \
'ptbc' or", " 'ptyj' or 'rb' or 'ss' value for scaler!"
    return x_vals


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


# Prepare data - calculate and scale biases.
# Extract X and Y values from train dataset for futher analyse
train_x_all = train.iloc[:,:-1].values
train_y_all = train.iloc[:,[-1]].values

# Encode  y to ohe
ohe = OneHotEncoder()
ohe.fit(train_y_all)
train_y_all_ohe = ohe.transform(train_y_all).toarray()

#  Find input biases for SS 
train_features = list(train.columns)[:-1]
input_biases=[]
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
    input_biases.append(bias)

# Scale input bias with standard scaler and standard scaler plus mms in -1,1
train_x_all_with_biases = np.append(np.array(input_biases).reshape(1,-1),
                                       train_x_all, axis=0)
train_x_all_with_biases_ss = npscaler(train_x_all_with_biases)
input_biases_ss = train_x_all_with_biases_ss[-1,:]

# Convert input biases to tf format
tf_input_bias = tf.keras.initializers.Constant(input_biases_ss)


# Find output bias
output_biases = []
for i in range (train_y_all_ohe.shape[1]):
    neg, pos = np.bincount(train_y_all_ohe[:,0].astype(int))/train_y_all_ohe[:,0].shape[0]
    output_biases.append(np.log(pos/neg))
output_biases = np.array(output_biases)

# Convert output biases to tf format
tf_output_bias = tf.keras.initializers.Constant(output_biases)
ds_train, ds_test = npds_train_test_split(train_x_all, train_y_all_ohe)

# 
x_train, x_test, y_train, y_test = train_test_split(train_x_all, train_y_all_ohe, 
                    test_size=0.2, stratify=train_y_all_ohe,
                    random_state=42)
x_train, x_test=npscaler(x_train), npscaler(x_test) 


# Find optimal model topology using search grids of Ray tune
  
def train_topology(config, shape_x, epochs, batch_size, x_train, y_train, 
                   x_test, y_test, input_bias, output_bias):
    # Define layers and neurons for each layer
    layers_no=config["layers"]
    layer_neurons = int(round(shape_x/(layers_no)))
    # Define metrics
    metrics_short = [
          tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
    
    # Create model
    model = Sequential()
    #input layer
    model.add(layers.Dense(units=shape_x, input_shape=(shape_x,), 
                           bias_initializer=input_bias,
                           activation=config["activators"]))
    # add hidden layer
    for i in range(layers_no):
            model.add(layers.Dense(units=layer_neurons, 
                                   activation=config["activators"]))
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=metrics_short)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({"mean_accuracy": "val_categorical_accuracy"})],
    )


def tune_topology(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        tune.with_parameters(train_topology,shape_x=train_x_all.shape[1], 
                             epochs=50, batch_size=512, x_train=x_train, 
                             y_train=y_train, x_test=x_test, y_test=y_test, 
                             input_bias=tf_input_bias, 
                             output_bias=tf_output_bias),
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={"mean_accuracy": 0.99, "training_iteration": num_training_iterations},
        num_samples=10,
        resources_per_trial={"cpu": 4, "gpu": 0},
        config={
            "threads": 4,
            "layers": tune.grid_search(list(range(2,9))),
            "activators": tune.grid_search(["relu", "selu", "elu"])
        },
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


# Find optimal kernel weights initializers
def train_weights_init(config, shape_x, layers_no, epochs, batch_size, x_train, y_train, 
                   x_test, y_test, input_bias, output_bias):
    # Define layers and neurons for each layer
    layers_no=layers_no
    layer_neurons = int(round(shape_x/(layers_no)))
    # Define metrics
    metrics_short = [
          tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
    
    # Create model
    model = Sequential()
    #input layer
    model.add(layers.Dense(units=shape_x, input_shape=(shape_x,),
                           kernel_initializer=config["weights_init"],
                           bias_initializer=input_bias, 
                           activation="relu"))
    # add hidden layer
    for i in range(layers_no):
            model.add(layers.Dense(units=layer_neurons,
                                   kernel_initializer=config["weights_init"],
                                   activation="relu"))
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=metrics_short)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({"mean_accuracy": "val_categorical_accuracy"})],
    )


def tune_weights_init(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        tune.with_parameters(train_weights_init,shape_x=train_x_all.shape[1], 
                             layers_no=4, epochs=50, batch_size=512, x_train=x_train, 
                             y_train=y_train, x_test=x_test, y_test=y_test, 
                             input_bias=tf_input_bias, 
                             output_bias=tf_output_bias),
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={"mean_accuracy": 0.99, "training_iteration": num_training_iterations},
        num_samples=10,
        resources_per_trial={"cpu": 4, "gpu": 0},
        config={
            "threads": 4,
            "weights_init": tune.grid_search(["RandomNormal", "RandomUniform",
                                             "TruncatedNormal", "GlorotNormal",
                                             "GlorotUniform", "HeNormal",
                                             "HeUniform", "VarianceScaling"])
        },
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


# Find optimal dropuot for each layer.. Catuion!!! Dropout look for  
# separately for each laers. If find for all layers it will very exaust.
def train_dropout(config, shape_x, epochs, batch_size, x_train, y_train, 
                   x_test, y_test, input_bias, output_bias):
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
    model.add(Dropout(rate=0.35))
    
    # First hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.05))
    
    # Second hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.0))
    
    # Trird layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.1))
    
    # Fourth layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=config["dr_4"]))
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=metrics_short)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({"mean_accuracy": "val_categorical_accuracy"})],
    )

# Find dropout rate for each layer separately by comment/uncomment required rows
def tune_dropout(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        tune.with_parameters(train_dropout,shape_x=train_x_all.shape[1], 
                             epochs=50, batch_size=512, x_train=x_train, 
                             y_train=y_train, x_test=x_test, y_test=y_test, 
                             input_bias=tf_input_bias, 
                             output_bias=tf_output_bias),
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={"mean_accuracy": 0.99, "training_iteration": num_training_iterations},
        num_samples=5,
        resources_per_trial={"cpu": 6, "gpu": 1},
        config={
            # "dr_in": tune.grid_search([0.0, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 
            #                           0.35, 0.4 , 0.45, 0.5 ]),
            # "dr_1": tune.grid_search([0.0, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 ,
            #                         0.35, 0.4 , 0.45, 0.5 ]),
            # "dr_2": tune.grid_search([0.0, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 
            #                          0.35, 0.4 , 0.45, 0.5 ])
            # "dr_3": tune.grid_search([0.0, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 
            #                          0.35, 0.4 , 0.45, 0.5 ]),
            "dr_4": tune.grid_search([0.0, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 
                                      0.35, 0.4 , 0.45, 0.5 ])
        },
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


# Add batch normalization and find optimal learn rate for ADAM
def train_adam(config, shape_x, epochs, batch_size, x_train, y_train, 
                   x_test, y_test, input_bias, output_bias):
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
    model.add(Dropout(rate=0.35))
    model.add(BatchNormalization())
    
    # First hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.05))
    model.add(BatchNormalization())
    
    # Second hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.0))
    model.add(BatchNormalization())
    
    # Trird layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())
    
    # Fourth layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=config["learn_rate"]),
        metrics=metrics_short)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({"mean_accuracy": "val_categorical_accuracy"})],
    )

# Find dropout rate for each layer separately by comment/uncomment required rows
def tune_adam(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        tune.with_parameters(train_adam,shape_x=train_x_all.shape[1], 
                             epochs=50, batch_size=512, x_train=x_train, 
                             y_train=y_train, x_test=x_test, y_test=y_test, 
                             input_bias=tf_input_bias, 
                             output_bias=tf_output_bias),
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={"mean_accuracy": 0.99, "training_iteration": num_training_iterations},
        num_samples=5,
        resources_per_trial={"cpu": 6, "gpu": 1},
        config={
             "learn_rate": tune.grid_search(list(np.logspace(-4,-2,17)))
        },
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


def train_sgd(config, shape_x, epochs, batch_size, x_train, y_train, 
                   x_test, y_test, input_bias, output_bias):
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
    model.add(Dropout(rate=0.35))
    model.add(BatchNormalization())
    
    # First hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.05))
    model.add(BatchNormalization())
    
    # Second hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.0))
    model.add(BatchNormalization())
    
    # Trird layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())
    
    # Fourth layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(learning_rate=config["learn_rate"]),
        metrics=metrics_short)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({"mean_accuracy": "val_categorical_accuracy"})],
    )

# Find dropout rate for each layer separately by comment/uncomment required rows
def tune_sgd(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        tune.with_parameters(train_sgd,shape_x=train_x_all.shape[1], 
                             epochs=50, batch_size=512, x_train=x_train, 
                             y_train=y_train, x_test=x_test, y_test=y_test, 
                             input_bias=tf_input_bias, 
                             output_bias=tf_output_bias),
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={"mean_accuracy": 0.99, "training_iteration": num_training_iterations},
        num_samples=5,
        resources_per_trial={"cpu": 6, "gpu": 1},
        config={
            # iterr 0 with best learining rate = 0.1 
            #"learn_rate": tune.grid_search(list(np.logspace(-3,-1,9)))
            # iter_1
            "learn_rate": tune.grid_search(list(np.linspace(0.1,0.4,7)))
        },
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


# define model with ADAM 
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



# Train model and return history df
def train_model(model, x_train, y_train_ohe, test_sz=0.2, batch_sz=1024, 
              stop_no=200) :
    callbacks = [EarlyStopping(monitor='val_categorical_accuracy',mode='max',
                                patience=stop_no,restore_best_weights=True)]

    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train_ohe,
                                                        test_size=test_sz,
                                                        stratify=y_train_ohe,
                                                        random_state=42)
    
    X_train, X_test = npscaler(X_train), npscaler(X_test)
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.shuffle(buffer_size=X_train.shape[0]).batch(batch_sz)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.shuffle(buffer_size=X_test.shape[0]).batch(batch_sz)
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

model_adam, model_adam_hist = train_model(clf_adam(), train_x_all, 
                                          train_y_all_ohe)


model_adam_hist.set_index("epochs",inplace=True)
model_adam_hist_loss = model_adam_hist[[ 'loss', 'val_loss']]

fig, ax = plt.subplots(figsize=(18,12))
ax = sns.lineplot(data=model_adam_hist_loss, ax = ax)

# Set major sticks
major_xticks = np.arange(0,model_adam_hist_loss.shape[0]+50, 50)
major_yticks = np.arange(0,model_adam_hist_loss.val_loss.max()+0.1, 0.1)

# Set minor sticks
minor_xticks = np.arange(0,model_adam_hist_loss.shape[0]+10, 10)
minor_yticks = np.arange(0,model_adam_hist_loss.val_loss.max()+0.04, 0.01)

# Define major and minor sticks
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

ax.set_yticks(major_yticks);
ax.set_yticks(minor_yticks, minor = True);

# ax labels
ax.set_ylabel('Loss', fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
ax.legend()
ax.set_title('Loss curve for ADAM clf with regularization and learning rate=\
5.623e-3.', fontsize=18,
             fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

# Evaluate model
# 3875/3875 [==============================] - 16s 4ms/step 
# - loss: 0.0228 - categorical_accuracy: 0.9952 
x_y = npds_train_test_split(train_x_all, train_y_all_ohe, 
                            test_ratio=None, batch_sz=32)

# Check accuracy of model
model_adam_evaluate = model_adam.evaluate(x_y, return_dict=True) 

# Recheck accurace

train_x_all_ss = npscaler(train_x_all)
y_pred = model_adam.predict(train_x_all_ss)
y_pred_int = np.where(y_pred < 0.5, 0, 1)

y_pred_acc = accuracy_score(train_y_all_ohe, y_pred_int)


# Train model with defalt learning rate without regularization
model_adam_def, model_adam_hist_def = train_model(clf_adam(learn_rate=0.001, 
             drop_out_in=0.0, drop_out_1=0.0, drop_out_2=0.0, drop_out_3=0.0,
             drop_out_4=0.0), train_x_all, train_y_all_ohe)

model_adam_hist_def.set_index("epochs",inplace=True)
model_adam_hist_loss_def = model_adam_hist_def[[ 'loss', 'val_loss']]

fig, ax = plt.subplots(figsize=(18,12))
ax = sns.lineplot(data=model_adam_hist_loss_def, ax=ax)

# Set major sticks
major_xticks = np.arange(0,model_adam_hist_loss_def.shape[0]+50, 50)
major_yticks = np.arange(0,model_adam_hist_loss_def.val_loss.max()+0.1, 0.1)

# Set minor sticks
minor_xticks = np.arange(0,model_adam_hist_loss_def.shape[0]+10, 10)
minor_yticks = np.arange(0,model_adam_hist_loss_def.val_loss.max()+0.04, 0.01)

# Define major and minor sticks
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

ax.set_yticks(major_yticks);
ax.set_yticks(minor_yticks, minor = True);

# ax labels
ax.set_ylabel('Loss', fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
ax.legend()
ax.set_title('Loss curve for ADAM clf without regularization and default\
learn rate=1e-3.', fontsize=18,
             fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

# Evaluate model without regularization and default learning rate
# 3875/3875 [==============================] - 17s 4ms/step 
# - loss: 0.0447 - categorical_accuracy: 0.9922 
model_adam_evaluate_def = model_adam_def.evaluate(x_y, return_dict=True) 



# Weighs decay - find optimal learn rate scheduler
def lr_scheduler(epochs, lr):
    if epochs <= 30:
        return lr
    else:
        return lr * math.exp(-0.09*epochs)

lr_callback = LearningRateScheduler(lr_scheduler)
es_callbacks = EarlyStopping(monitor='categorical_accuracy',mode='max',
                            patience=30,restore_best_weights=True)


# Define train and test tf_datasets
ds_train, ds_test = npds_train_test_split(train_x_all, train_y_all_ohe)

model = clf_adam(shape_x=train_x_all.shape[1], learn_rate=0.005623, 
             drop_out_in=0.35, drop_out_1=0.05, drop_out_2=0.0, drop_out_3=0.1,
             drop_out_4=0.1, input_bias=tf_input_bias, 
             output_bias=tf_output_bias)
lr_hist = model.fit(ds_train,  epochs=1000, batch_size=512,
                    callbacks=[es_callbacks, lr_callback], 
                    validation_data=ds_test)

lr_hist_df = pd.DataFrame(data=lr_hist.history)
lr_hist_df["epochs"]=lr_hist_df.index + 1

model_lr_adam_evaluate = model.evaluate(x_y, return_dict=True) 

lr_hist_df.set_index("epochs",inplace=True)
lr_hist_df_loss = lr_hist_df[[ 'loss', 'val_loss']]

fig, ax = plt.subplots(figsize=(18,12))
ax = sns.lineplot(data=lr_hist_df_loss, ax=ax)

# Set major sticks
major_xticks = np.arange(0,lr_hist_df.shape[0]+50, 50)
major_yticks = np.arange(0,lr_hist_df.val_loss.max()+0.1, 0.1)

# Set minor sticks
minor_xticks = np.arange(0,lr_hist_df.shape[0]+10, 10)
minor_yticks = np.arange(0,lr_hist_df.val_loss.max()+0.04, 0.01)

# Define major and minor sticks
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor = True)

ax.set_yticks(major_yticks);
ax.set_yticks(minor_yticks, minor = True);

# ax labels
ax.set_ylabel('Loss', fontsize=12, fontweight="bold")
ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
ax.legend()
ax.set_title('Loss curve for ADAM clf without regularization and default\
learn rate=1e-3.', fontsize=18,
             fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using " "Ray Client.",
    )
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=4)
    elif args.server_address:
        ray.init(f"ray://{args.server_address}")

    # Find optimal network topology    
    tune_analysis = tune_topology(num_training_iterations=5 if args.smoke_test else 10)
    # 2022-04-29 23:45:17,043	INFO tune.py:701 -- Total run time: 3283.41 seconds 
    # (3283.17 seconds for the tuning loop).
    # Best hyperparameters found were:  {'threads': 4, 'layers': 4, 'activators': 'relu'}
    
    # Find optimal kernel weights initializer
    tune_weights_analysis = tune_weights_init(num_training_iterations=3 if args.smoke_test else 6)
    # 2022-04-30 09:36:51,822	INFO tune.py:701 -- 
    # Total run time: 724.29 seconds (724.10 seconds for the tuning loop).
    # Best hyperparameters found were:  {'threads': 4, 'weights_init': 'GlorotUniform'}
    
    # Find optimal dropout for each layers
    tune_drop_in_analysis = tune_dropout(num_training_iterations=1 if args.smoke_test else 300)
    tune_drop_in_analysis_df =  tune_drop_in_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 10:17:17,621	INFO tune.py:701 -- Total run time: 3063.53
    # seconds (3063.37 seconds for the tuning loop).
    # Best hyperparameters found were:  {'threads': 4, 'dr_in': 0.35}
    
    tune_drop_1_analysis = tune_dropout(num_training_iterations=1 if 
                                         args.smoke_test else 300)
    tune_drop_1_analysis_df = tune_drop_1_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 12:15:34,072	INFO tune.py:701 -- Total run time: 2862.45 seconds 
    # (2862.30 seconds for the tuning loop).
    # Best hyperparameters found were:  {'threads': 4, 'dr_1': 0.05}
    
    tune_drop_2_analysis = tune_dropout(num_training_iterations=10 if 
                                         args.smoke_test else 300)
    tune_drop_2_analysis_df = tune_drop_2_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 12:48:15,482	INFO tune.py:701 -- Total run time: 1569.47 
    # seconds (1569.30 seconds for the tuning loop).
    # Best hyperparameters found were:  {'dr_2': 0.0}
    
    tune_drop_3_analysis = tune_dropout(num_training_iterations=10 if 
                                         args.smoke_test else 300)
    tune_drop_3_analysis_df = tune_drop_3_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 13:18:30,838	INFO tune.py:701 -- Total run time: 1478.00 seconds 
    # (1477.82 seconds for the tuning loop).
    # Best hyperparameters found were:  {'dr_3': 0.1}
    
    tune_drop_4_analysis = tune_dropout(num_training_iterations=10 if 
                                         args.smoke_test else 300)
    tune_drop_4_analysis_df = tune_drop_4_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 13:57:16,328	INFO tune.py:701 -- Total run time: 1587.66 
    # seconds (1587.46 seconds for the tuning loop).
    # Best hyperparameters found were:  {'dr_4': 0.1}
    
    tune_adam_analysis = tune_adam(num_training_iterations=10 if 
                                         args.smoke_test else 300)
    tune_adam_analysis_df = tune_adam_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 16:54:18,924	INFO tune.py:701 -- Total run time: 6200.02 
    # seconds (6199.83 seconds for the tuning loop).
    # Best hyperparameters found were:  {'learn_rate': 0.005623413251903491}
    
    # iteration 0
    tune_sgd_analysis = tune_sgd(num_training_iterations=10 if 
                                         args.smoke_test else 300)
    tune_sgd_analysis_0_df = tune_sgd_analysis.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 18:32:51,167	INFO tune.py:701 
    # Total run time: 3376.49 seconds (3376.32 seconds for the tuning loop).
    # Best hyperparameters found were:  {'learn_rate': 0.1}
    
    # iter1 
    tune_sgd_analysis_1 = tune_sgd(num_training_iterations=10 if 
                                         args.smoke_test else 300)
    tune_sgd_analysis_1_df = tune_sgd_analysis_1.dataframe(
        metric="mean_accuracy", mode="max")
    # 2022-05-01 20:38:43,074	INFO tune.py:701 
    # -- Total run time: 2335.18 seconds (2335.00 seconds for the tuning loop).
    # Best hyperparameters found were:  {'learn_rate': 0.25}
    
    
    
    