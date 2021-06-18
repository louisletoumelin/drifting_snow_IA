from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_data(prm):
    """Load data"""
    D17_observations = pd.read_pickle(prm["path_data"] + "D17_observations.pkl")
    D17_total = pd.read_pickle(prm["path_data"] + "D17_total.pkl")
    MAR = pd.read_pickle(prm["path_data"] + "MAR_ERA5_22_Mars_2021.pkl")
    MAR_no_BS_ERA5 = pd.read_pickle(prm["path_data"] + "MAR_no_BS_ERA5_22_Mars_2021.pkl")
    return D17_observations, D17_total, MAR, MAR_no_BS_ERA5


def degree_to_kelvin(df, variables):
    """Temperatures in degrees"""
    for variable in variables:
        df[variable] = df[variable] + 273
    return df


def negative_to_nans(df, variables):
    """Keep only RH > 0"""
    for variable in variables:
        df[variable][df[variable] < 0] = np.nan
    return df


def remove_high_FC_values(D17_total):
    too_high_FC1 = D17_total["FC1"] > 600
    D17_total["FC1"][too_high_FC1] = np.nan
    return (D17_total[D17_total["FC1"].notna()])


def preprocess_predict_FC2(D17_total, all_var):
    # Training set before 2018
    after_2014 = (D17_total.index.year >= 2014)
    before_2017 = (D17_total.index.year <= 2017)
    Dataset_FC2 = D17_total[all_var][after_2014 & before_2017].dropna()
    X_train_FC2 = Dataset_FC2[all_var[:-1]]
    y_train_FC2 = Dataset_FC2[all_var[-1]]

    # Testing set in 2018
    in_2018 = (D17_total.index.year == 2018)
    Dataset_FC2 = D17_total[all_var][in_2018].dropna()
    X_test_FC2 = Dataset_FC2[all_var[:-1]]
    y_test_FC2 = Dataset_FC2[all_var[-1]]

    # Prediction before 2013
    before_2013 = (D17_total.index.year <= 2013)
    X_predict_FC2 = D17_total[all_var[:-1]][before_2013]

    return [X_train_FC2, y_train_FC2, X_test_FC2, y_test_FC2, y_test_FC2, X_predict_FC2]


def predict_FC2(X_train_FC2, y_train_FC2, X_test_FC2, y_test_FC2, X_predict_FC2):
    # Train with a Random Forest
    rnd_clf = RandomForestRegressor(n_estimators=10)
    rnd_clf.fit(X_train_FC2, y_train_FC2)

    # Test
    y_test_rf_FC2 = rnd_clf.predict(X_test_FC2)
    print(y_test_FC2.shape, y_test_rf_FC2.shape)
    print('RMSE FC2 prediction: \n')
    print(mean_squared_error(y_test_FC2, y_test_rf_FC2) ** 0.5)

    # Prediction
    y_pred_FC2 = rnd_clf.predict(X_predict_FC2)

    # Plot test set
    """
    ax = plt.gca()
    y_test_FC2.plot(ax=ax, marker='x')
    y_test_rf_FC2 = pd.DataFrame(y_test_rf_FC2)
    y_test_rf_FC2.index = y_test_FC2.index
    y_test_rf_FC2.plot(ax=ax, marker='d')
    plt.legend(())
    plt.title("Learning to reconstruct FC2")

    # Plot learning set
    ax = plt.gca()
    y_train_FC2.plot(ax=ax, marker='x')
    y_train_rf_FC2 = pd.DataFrame(rnd_clf.predict(X_train_FC2))
    y_train_rf_FC2.index = y_train_FC2.index
    y_train_rf_FC2.plot(ax=ax, marker='d')

    # Plot prediction
    y_pred_FC2 = pd.DataFrame(y_pred_FC2)
    y_pred_FC2.index = X_predict_FC2.index
    y_pred_FC2.plot(ax=ax, marker='d')
    plt.legend(('Observed test', 'Predicted test', 'Observed learning', 'Predicted learning', "Reconstructed FC2"))
    """
    return y_pred_FC2


def integrate_FC2_dataset(D17_total, y_pred_FC2):
    # Integrate FC2
    #D17_total["FC2"].plot()
    y_pred_FC2 = pd.DataFrame(y_pred_FC2)
    y_pred_FC2.columns = ["FC2"]
    D17_total.update(y_pred_FC2)
    #D17_total["FC2"].plot()
    return D17_total


def calculate_FC(D17_total):
    D17_total['hFC1'][D17_total['hFC1'] < 0] = 0
    D17_total['hFC1'][D17_total['hFC1'] > 1] = 1

    D17_total['FC_2'] = (D17_total['FC1'] + D17_total['FC2']) / 2
    D17_total['FC_2'][D17_total['hFC1'] < 1] = (D17_total['FC1'] * D17_total['hFC1'] + D17_total['FC2']) / (
            D17_total['hFC1'] + 1)

    return D17_total


def delete_some_constant_values(D17_observations, D17_total):

    # 25/12/2015
    # FC = 314.48 for 7 hours on the 25th of December 2012
    year_2015 = (D17_observations.index.year == 2015)
    month_12 = (D17_observations.index.month == 12)
    day_after_25 = (D17_observations.index.day >= 25)
    day_before_26 = (D17_observations.index.day <= 26)
    D17_observations['FC_2'][year_2015 & month_12 & day_after_25 & day_after_25 & day_before_26] = np.nan

    year_2015 = (D17_total.index.year == 2015)
    month_12 = (D17_total.index.month == 12)
    day_after_25 = (D17_total.index.day >= 25)
    day_before_26 = (D17_total.index.day <= 26)
    D17_total['FC_2'][year_2015 & month_12 & day_after_25 & day_after_25 & day_before_26] = np.nan

    # 06/01/2018
    # FC = for 13 hours on the 3rd of December 2018
    year_2018 = (D17_total.index.year == 2018)
    month_1 = (D17_total.index.month == 1)
    day_after_3 = (D17_total.index.day >= 3)
    day_before_4 = (D17_total.index.day <= 4)
    D17_total['FC_2'][year_2018 & month_1 & day_after_3 & day_before_4] = np.nan

    # FC = 183.06 for 18 hours on the 6th of December 2018
    year_2018 = (D17_total.index.year == 2018)
    month_1 = (D17_total.index.month == 1)
    day_after_6 = (D17_total.index.day >= 6)
    day_before_8 = (D17_total.index.day <= 8)
    D17_total['FC_2'][year_2018 & month_1 & day_after_6 & day_before_8] = np.nan

    return D17_observations, D17_total


def merge_dataset(D17_observations, D17_total):
    D17_total = D17_total.merge(D17_observations, left_index=True, right_index=True)
    return D17_total


def add_snowfall_from_MAR(D17_total, MAR, prm):
    if prm["GPU"]:
        SF = pd.read_csv(prm["path_data"] + "MAR_no_BS_ERA5_SF.csv")
        SF.columns = ["index", "SF"]
        SF.index = SF["index"]
        SF.index = pd.to_datetime(SF.index)
        D17_total["SF"] = np.nan
        filter_SF = SF.index.isin(D17_total.index)
        filter_D17_total = D17_total.index.isin(SF.index)
        D17_total["SF"][filter_D17_total] = SF["SF"][filter_SF].values
    else:
        print(pd.__version__)
        D17_total["SF"] = np.nan
        filter_MAR = MAR.index.isin(D17_total.index)
        filter_D17_total = D17_total.index.isin(MAR.index)
        D17_total["SF"][filter_D17_total] = MAR["SF"][filter_MAR].values
    return D17_total


def temporal_gradient_snow_height(D17_total):
    var_height = ['zT1', 'zT2', 'zT3', 'zT4', 'zT5', 'zT6']
    for variable in var_height:
        D17_total['gradient_' + variable] = D17_total[variable].diff()
        D17_total['gradient_' + variable][(abs(D17_total['gradient_' + variable]) > 0.10)] = np.nan
    return D17_total


def vertical_gradients(D17_total):
    D17_total['vert_grad_RH'] = D17_total['RH1'] - D17_total['RH6']
    D17_total['vert_grad_T'] = D17_total['T1'] - D17_total['T6']
    D17_total['vert_grad_U'] = D17_total['U1'] - D17_total['U6']
    return D17_total


def select_variables_and_dropna(D17_total, variables_training):
    Dataset = D17_total[variables_training]
    Dataset = Dataset.dropna()
    return Dataset


def split_train_test_valid(Dataset, variables=None, label="FC_2_y"):
    year_max = 2018

    X_train = Dataset[variables[:-1]][(Dataset.index.year < year_max)].values
    X_test = Dataset[variables[:-1]][(Dataset.index.year == year_max)].values

    y_train = Dataset[label][(Dataset.index.year < year_max)].values
    y_test = Dataset[label][(Dataset.index.year == year_max)].values

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.001)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def scale(X_train, X_valid, X_test):
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return X_train, X_valid, X_test


def train_test_split_scale(test_year, Dataset, label="FC_2_y"):
    # Training and Testing sets
    variables = Dataset.columns
    X_train = Dataset[variables[:-1]][(Dataset.index.year != test_year)].values
    X_test = Dataset[variables[:-1]][(Dataset.index.year == test_year)].values

    y_train = Dataset[label][(Dataset.index.year != test_year)].values
    y_test = Dataset[label][(Dataset.index.year == test_year)].values
    y_test_index = Dataset[label][(Dataset.index.year == test_year)].index

    # Validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.001)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, y_test_index
