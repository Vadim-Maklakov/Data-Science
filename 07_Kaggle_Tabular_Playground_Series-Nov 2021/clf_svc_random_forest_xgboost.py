#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
https://www.kaggle.com/c/tabular-playground-series-nov-2021/data

"""
import numpy as np
import pandas as pd
import time

# Import EDA function

from eda_stat_func import mutual_clf_fs
from eda_stat_func import best_clf_est_cv

# Temp imports


# import xgboost as xgb

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import KFold


# 1. Read train datasatet 
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")

# drop id column
train = train_raw.iloc[:, 1:].copy()
test = test_raw.iloc[:, 1:].copy()
 

# 2. Estimate best features for train dataset
t = time.process_time()
fs_mut_clf, mut_clf_qt_desc, mut_clf_pw_yj_desc = mutual_clf_fs(train)
fs_calc_time = time.process_time() - t

# Select  statistical significant features  
fs_lst =  list(fs_mut_clf.fs_name.values)[1:-1]
# add `target` column as y 
fs_lst.append(list(fs_mut_clf.fs_name.values)[0])


# Select test train dataset
train_test_ds = train[fs_lst].iloc[:2500, :].copy()

pipe_qt = [('qt', QuantileTransformer(output_distribution='normal'))] 
pipe_qt_std = [('std', StandardScaler()),
               ('qt', QuantileTransformer(output_distribution='normal'))] 
pipe_qt_mms = [('mms', MinMaxScaler()), 
               ('qt', QuantileTransformer(output_distribution='normal'))]
pipe_qt_rbs = [('rbs', RobustScaler()), 
               ('qt', QuantileTransformer(output_distribution='normal'))]

pipe_mms_pw_bc = [('mms', MinMaxScaler(feature_range=(1, 2))), 
                ("pw_bc", PowerTransformer(method='box-cox'))]
pipe_std = [('std', StandardScaler())] 

t1 = time.process_time()
# for debug
# pipe_list = pipe_qt[:]
# pipe_pw_yj = [("pw_yj", PowerTransformer(method='yeo-johnson'))]

clf_stats_qt = best_clf_est_cv(train_test_ds, pipe_qt)
clf_stats_qt_std = best_clf_est_cv(train_test_ds, pipe_qt_std)
clf_stats_qt_mms = best_clf_est_cv(train_test_ds, pipe_qt_mms)
clf_stats_qt_rbs = best_clf_est_cv(train_test_ds, pipe_qt_rbs)
clf_stats_mms_pw_bc = best_clf_est_cv(train_test_ds, pipe_mms_pw_bc)
clf_stats_std = best_clf_est_cv(train_test_ds, pipe_std)
clf_best_calc_time = time.process_time() - t1


