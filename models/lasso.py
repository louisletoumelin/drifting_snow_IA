import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


def lasso_regr(X_train, y_train, X_test, y_test, results):

    # Fit
    lasso_reg = Lasso(alpha=0.3)
    lasso_reg.fit(X_train, y_train)

    # Predict
    y_pred = lasso_reg.predict(X_test)

    # Cleaning prediction
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    y_pred[y_pred<0] = 0
    y_pred = y_pred-y_pred.min()

    corr_coeff = pd.concat([y_pred, y_test], axis=1).corr().iloc[0,1]
    rmse = mean_squared_error(y_test, y_pred)**(0.5)
    bias = np.mean(y_pred - y_test)

    results["lasso"]["corr_coeff"].append(corr_coeff)
    results["lasso"]["rmse"].append(rmse)
    results["lasso"]["bias"].append(bias)
    results["lasso"]["y_pred"].append(y_pred)

    return(results)