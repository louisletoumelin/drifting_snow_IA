from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from preprocess.preprocess import *
from models.random_forest import *
from cross_validate import *
from prm.prm import *
from models.lasso import *
from models.dnn import *
from utils.utils import *

pd.options.mode.chained_assignment = None

prm = PRM()

config_CPU_GPU(prm)

# Load and preprocess data
D17_observations, D17_total, MAR, MAR_no_BS_ERA5 = load_data(prm)
D17_total = degree_to_kelvin(D17_total, ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
D17_observations = degree_to_kelvin(D17_observations, ['T2m'])
D17_total = negative_to_nans(D17_total, ['RH1', 'RH2', 'RH3', 'RH4', 'RH5', 'RH6'])
D17_total = remove_high_FC_values(D17_total)

# Predict FC2
predict_variable_FC2 = ['FC1', 'RH1', 'U1', 'FC2']
list_preprocess = preprocess_predict_FC2(D17_total, all_var=predict_variable_FC2)
X_train_FC2, y_train_FC2, X_test_FC2, y_test_FC2, y_test_FC2, X_predict_FC2 = list_preprocess
y_pred_FC2 = predict_FC2(X_train_FC2, y_train_FC2, X_test_FC2, y_test_FC2, X_predict_FC2)
D17_total = integrate_FC2_dataset(D17_total, y_pred_FC2)

# Preprocess FC
D17_total = calculate_FC(D17_total)
D17_observations, D17_total = delete_some_constant_values(D17_observations, D17_total)

# Merge datasets
D17_total = merge_dataset(D17_observations, D17_total)

# Add new variables
D17_total = add_snowfall_from_MAR(D17_total, MAR_no_BS_ERA5, prm)
D17_total = temporal_gradient_snow_height(D17_total)
D17_total = vertical_gradients(D17_total)

"""
['DD_x', 'FC1_x', 'FC2_x', 'MM_x', 'RH1', 'RH2', 'RH3', 'RH4', 'RH5',
    'RH6', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'U1', 'U2', 'U3', 'U4', 'U5',
    'U6', 'YYYY_x', 'hFC1', 'hFC2', 'hh_x', 'mm_x', 'zT1', 'zT2', 'zT3',
    'zT4', 'zT5', 'zT6', 'zU1', 'zU2', 'zU3', 'zU4', 'zU5', 'zU6', 'FC_2_x',
    'YYYY_y', 'MM_y', 'DD_y', 'hh_y', 'mm_y', 'U2m', 'zU', 'T2m', 'RH2m',
    'zT', 'FC1_y', 'FC2_y', 'zFC1', 'SP', 'SWU', 'SWD', 'LWU', 'LWD',
    'FC_3', 'zFC1_2', 'FC_2_y', 'gradient_zT1', 'gradient_zT2',
    'gradient_zT3', 'gradient_zT4', 'gradient_zT5', 'gradient_zT6',
    'vert_grad_RH', 'vert_grad_T', 'vert_grad_U']
"""

"""
variables_training = ["U2m", "RH2m", "T2m", "FC_2_y"]
results_1 = cross_validate(D17_total, MAR, variables_training)
print_results(results_1, variables_training)
pd.to_pickle(results_1, "result_1.pkl")
"""
# Additional sensors
variables_training = ["U2m", "RH2m", "T2m", "RH3", "RH4", "RH5", "RH6", "T2", "T3", "T4", "T5", "FC_2_y"]
results_2 = cross_validate(D17_total, MAR, variables_training)
print_results(results_2, variables_training)
pd.to_pickle(results_2, "results_2.pkl")
"""

# Gradients height
variables_training = ["RH2m", "T2m", "U2m", 'gradient_zT1', 'gradient_zT2',
    'gradient_zT3', 'gradient_zT4', 'gradient_zT5', 'gradient_zT6', "FC_2_y"]
results_3 = cross_validate(D17_total, MAR, variables_training)
print_results(results_3, variables_training)
pd.to_pickle(results_3, "results_3.pkl")


# Radiative measurements
variables_training = ["RH2m", "T2m", "U2m",  'SWU', 'SWD', 'LWU', 'LWD', "FC_2_y"]
results_4 = cross_validate(D17_total, MAR, variables_training)
print_results(results_4, variables_training)
pd.to_pickle(results_4, "results_4.pkl")

# Wind only
variables_training = ["U2m", "FC_2_y"]
results_5 = cross_validate(D17_total, MAR, variables_training)
print_results(results_5, variables_training)
pd.to_pickle(results_5, "results_5.pkl")

# Snowfalls from MAR
variables_training = ["U2m", "RH2m", "T2m", "SF", "FC_2_y"]
results_6 = cross_validate(D17_total, MAR, variables_training)
print_results(results_6, variables_training)
pd.to_pickle(results_6, "results_6.pkl")

# All variables
variables_training = ["U2m", "RH2m", "T2m", "RH3", "RH4", "RH5", "RH6", "T2", "T3", "T4", "T5", 'gradient_zT1', 'gradient_zT2',
    'gradient_zT3', 'gradient_zT4', 'gradient_zT5', 'gradient_zT6', 'vert_grad_RH', 'vert_grad_T', 'SWU', 'SWD', 'LWU', 'LWD', "FC_2_y"]
results_7 = cross_validate(D17_total, MAR, variables_training)
print_results(results_7, variables_training)
pd.to_pickle(results_7, "results_7.pkl")
"""
