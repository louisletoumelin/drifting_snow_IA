import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor


def transformer_reg(X_train, y_train, X_test, y_test, results):

    y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

    transformer = TabNetRegressor()
    transformer.fit(X_train, y_train)

    # Predict
    y_pred = transformer.predict(X_test)

    # Cleaning prediction
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    y_pred[y_pred<0] = 0
    y_pred = y_pred-y_pred.min()

    corr_coeff = pd.concat([y_pred, y_test], axis=1).corr().iloc[0,1]
    rmse = mean_squared_error(y_test, y_pred)**(0.5)
    bias = np.mean(y_pred - y_test)

    results["transformer"]["corr_coeff"].append(corr_coeff)
    results["transformer"]["rmse"].append(rmse)
    results["transformer"]["bias"].append(bias)
    results["transformer"]["y_pred"].append(y_pred)

    return(results)
