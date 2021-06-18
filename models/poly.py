import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso


def poly_regr(X_train, y_train, X_test, y_test, results):

    # Prepare polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)

    # Fit
    lasso_reg = Lasso(alpha=0.3)
    lasso_reg.fit(X_poly, y_train)

    # Predict
    X_test_poly = poly_features.fit_transform(X_test)
    y_pred = lasso_reg.predict(X_test_poly)

    # Cleaning prediction
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    y_pred[y_pred<0] = 0
    y_pred = y_pred-y_pred.min()

    corr_coeff = pd.concat([y_pred, y_test], axis=1).corr().iloc[0,1]
    rmse = mean_squared_error(y_test, y_pred)**(0.5)
    bias = np.mean(y_pred - y_test)

    results["poly"]["corr_coeff"].append(corr_coeff)
    results["poly"]["rmse"].append(rmse)
    results["poly"]["bias"].append(bias)
    results["poly"]["y_pred"].append(y_pred)

    return(results)
