from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def best_random_forest():
    rnd_clf = RandomForestRegressor(bootstrap=True,
                                    max_depth=130,
                                    max_features='auto',
                                    min_samples_leaf=1,
                                    min_samples_split=8,
                                    n_estimators=1166,
                                    n_jobs=-1)
    return (rnd_clf)


def plot_importance_feature(Dataset, rf):
    for name, score in zip(Dataset.columns, rf.feature_importances_):
        print(name, np.round(score, 2))

    plt.figure(figsize=(15, 15))
    ind = np.arange(len(rf.feature_importances_))
    plt.bar(ind, rf.feature_importances_)
    plt.xticks(ind, list(Dataset.columns))


def random_forest_regr(X_train, y_train, X_test, y_test, results):
    # Fit
    rnd_clf = best_random_forest()
    rnd_clf.fit(X_train, y_train)

    # Predict
    y_pred = rnd_clf.predict(X_test)

    # Cleaning prediction
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)

    y_pred[y_pred < 0] = 0
    y_pred = y_pred - y_pred.min()

    corr_coeff = pd.concat([y_pred, y_test], axis=1).corr().iloc[0, 1]
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    bias = np.mean(y_pred - y_test)

    results["rf"]["corr_coeff"].append(corr_coeff)
    results["rf"]["rmse"].append(rmse)
    results["rf"]["bias"].append(bias)
    results["rf"]["y_pred"].append(y_pred)

    return results
