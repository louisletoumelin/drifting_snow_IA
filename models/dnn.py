from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import keras

from prm.prm import *


def choose_optimizer(name, learning_rate):
    """
      _____________________________________________________________
      _____________________________________________________________
      _______ Function use to CV search the best optimizer ________
      __________Input = Name, Leaning rate_________________________
      _____________________________________________________________
    """

    if name == "SGD":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)

    if name == "RMSprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)

    if name == "Adagrad":
        return keras.optimizers.Adagrad(learning_rate=learning_rate)

    if name == "Adadelta":
        return keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95)

    if name == "Adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    if name == "Adamax":
        return keras.optimizers.Adamax(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    if name == "Nadam":
        return keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)


def build_model(X_train, y_train, X_valid_scaled, y_valid, params):
    """
    _____________________________________________________________
    _____________________________________________________________
    _________________     Classic        ________________________
    _____________________________________________________________
    _____________________________________________________________
    """

    if params["architecture"] == 1:

        # ______________   Building model   ___________________________
        model = keras.models.Sequential()
        options = {"input_shape": X_train.shape[1:]}

        # ______________   Adding layers   ____________________________
        for layer in range(params['n_layers']):
            model.add(keras.layers.Dense(params['n_neurons'],
                                         activation=params['activation'],
                                         kernel_initializer=params['kernel_initializer'],
                                         **options))
            model.add(keras.layers.Dropout(0.2))
            options = {}

        # ______________   Output layer   _____________________________
        model.add(keras.layers.Dense(1, kernel_initializer=params['kernel_initializer'], **options))

        # ______________   Choosing optimizer   _______________________
        optimizer = choose_optimizer(params["optimizer"], params["learning_rate"])

        # __________________   Compiling   ____________________________
        model.compile(loss=params["loss"], optimizer=optimizer)

    ''' 
    _____________________________________________________________
    _____________________________________________________________
    _________________     Concatenate        ____________________
    _____________________________________________________________
    _____________________________________________________________
    '''

    if params["architecture"] == 2:

        # ______________   Building model   ___________________________
        model = keras.models.Sequential()
        input = keras.layers.Input(shape=X_train.shape[1:])

        # ______________   Adding layers   ____________________________
        hidden = keras.layers.Dense(params['n_neurons'], activation=params['activation'],
                                    kernel_initializer=params['kernel_initializer'])(input)
        for layer in range(params['n_layers'] - 1):
            hidden = keras.layers.Dense(params['n_neurons'], activation=params['activation'],
                                        kernel_initializer=params['kernel_initializer'])(hidden)

        # ______________   Concatenate   ____________________________
        concat = keras.layers.Concatenate()([input, hidden])

        # ______________   Output layer   _____________________________
        output = keras.layers.Dense(1)(concat)
        model = keras.models.Model(inputs=[input], outputs=[output])

        # ______________   Choosing optimizer   _______________________
        optimizer = choose_optimizer(params["optimizer"], params["learning_rate"])

        # __________________   Compiling   ____________________________
        model.compile(loss=params["loss"], optimizer=optimizer)

    ''' 
    _____________________________________________________________
    _____________________________________________________________
    _________________  Batch Normalization   ____________________
    ___for deeper networks it can make a tremendous difference___
    _____________________________________________________________
    '''

    if params["architecture"] == 3:
        # ______________   Building model   ___________________________
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=X_train.shape[1:]),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(300, activation=params['activation'], kernel_initializer=params['kernel_initializer']),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(200, activation=params['activation'], kernel_initializer=params['kernel_initializer']),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(100, activation=params['activation'], kernel_initializer=params['kernel_initializer']),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(5, activation=params['activation'], kernel_initializer=params['kernel_initializer']),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1)])

        # ______________   Choosing optimizer   _______________________
        optimizer = choose_optimizer(params["optimizer"], params["learning_rate"])

        # __________________   Compiling   ____________________________
        model.compile(loss=params["loss"], optimizer=optimizer)

    return model


def dnn_regr(X_train, y_train, X_test, y_test, X_valid, y_valid, results):
    # Fit
    return_best_dnn_fit = True
    best_dnn = build_model(X_train, y_train, X_valid, y_valid, PRM())
    best_dnn.fit(X_train, y_train, PRM()['epochs'], verbose=0)

    # Predict
    y_pred = best_dnn.predict(X_test)

    # Cleaning prediction
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    y_pred[y_pred < 0] = 0

    corr_coeff = pd.concat([y_pred, y_test], axis=1).corr().iloc[0, 1]
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    bias = np.mean(y_pred - y_test)

    results["dnn"]["corr_coeff"].append(corr_coeff)
    results["dnn"]["rmse"].append(rmse)
    results["dnn"]["bias"].append(bias)
    results["dnn"]["y_pred"].append(y_pred)

    return results
