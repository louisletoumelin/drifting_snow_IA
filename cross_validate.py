from collections import defaultdict
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

from models.lasso import *
from models.poly import *
from models.random_forest import *
from models.dnn import *
from models.transformer import *


def prepare_Dataset(D17_total, input_variables, test_year, label="FC_2_y"):

    # Delete NaNs
    Dataset = D17_total[input_variables]
    Dataset = Dataset.dropna()

    # Training and Testing sets
    training_years = (Dataset.index.year >= 2013) & (Dataset.index.year != test_year)
    test_year = (Dataset.index.year == test_year)

    X_train = Dataset[input_variables[:-1]][training_years].values
    X_test = Dataset[input_variables[:-1]][test_year].values

    y_train = Dataset[label][training_years].values
    y_test = Dataset[label][test_year].values
    y_test_index = Dataset[label][test_year].index

    # Validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.001)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return(X_train, y_train, X_test, y_test, y_test_index, X_valid, y_valid)


def cross_validate(D17_total, MAR, input_variables):
    list_years = [2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018]

    # Initialization mse list
    results = defaultdict(lambda: defaultdict(dict))

    for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar", "ensemble"]:
        for metric in ["rmse", "corr_coeff", "bias", "y_pred"]:
            results[algo][metric] = []
    results["index"] = []

    for test_year in list_years:
        start_time = time.time()
        X_train, y_train, X_test, y_test, y_test_index, X_valid, y_valid = prepare_Dataset(D17_total, input_variables, test_year)
        results["index"].append(y_test_index)

        print('\n' + str(test_year))

        # Lasso
        results = lasso_regr(X_train, y_train, X_test, y_test, results)
        print('Lasso finished')
        print("--- %s seconds for Lasso" % (time.time() - start_time))

        results = poly_regr(X_train, y_train, X_test, y_test, results)
        print('Poly finished')
        print("--- %s seconds for Poly" % (time.time() - start_time))

        results = random_forest_regr(X_train, y_train, X_test, y_test, results)
        print('Random forest finished')
        print("--- %s seconds for Forest" % (time.time() - start_time))

        results = dnn_regr(X_train, y_train, X_test, y_test, X_valid, y_valid, results)
        print('Deep neural network finished')
        print("--- %s seconds for DNN" % (time.time() - start_time))

        results = transformer_reg(X_train, y_train, X_test, y_test, results)
        print('Transformer finished')
        print("--- %s seconds for transformer" % (time.time() - start_time))

        y_test = pd.DataFrame(y_test)
        filter_1 = MAR.index.isin(D17_total['FC_2_y'][(D17_total.index.year == test_year)].index)
        filter_2 = MAR.index.isin(y_test_index)
        y_test_MAR = MAR['FC'][filter_1 & filter_2]
        y_test_MAR = y_test_MAR.to_frame()
        y_test.index = y_test_MAR.index

        # RMSE
        results["mar"]["corr_coeff"].append(pd.concat([y_test_MAR, y_test], axis=1).corr().iloc[0, 1])
        results["mar"]["rmse"].append(mean_squared_error(y_test, y_test_MAR) ** (0.5))
        results["mar"]["bias"].append(y_test_MAR.mean() - y_test.mean())
        results["mar"]["y_pred"].append(y_test_MAR)

        print("--- %s seconds for ---" % (time.time() - start_time))

    return(results)

