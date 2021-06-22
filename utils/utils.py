import numpy as np


def connect_GPU_to_horovod():
    import horovod.tensorflow.keras as hvd
    import tensorflow as tf
    tf.keras.backend.clear_session()
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def config_CPU_GPU(prm):
    connect_GPU_to_horovod() if prm["GPU"] else None


def print_results(results, variables_training):
    print(variables_training)
    print("\nrmse")
    for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar", "ensemble"]:
        print(algo, np.nanmean(results[algo]["rmse"]))

    print("\ncorr_coeff")
    for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar", "ensemble"]:
        print(algo, np.nanmean(results[algo]["corr_coeff"]))

    print("\nbias")
    for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar", "ensemble"]:
        print(algo, np.nanmean(results[algo]["bias"]))


"""
import os

from metrics.confusion_matrix import *

result=results_5
name="results_5"
for idx_year, year in enumerate([2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018]):
    for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar"]:
        variable = result[algo]["y_pred"][idx_year]
        variable.index = result["index"][idx_year]
        variable.to_pickle("results/" + name+"_"+str(int(year))+"_"+algo+".pkl")
    
df_all_predictions = pd.DataFrame()
for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar"]:
    all_predictions = []
    for idx_year, year in enumerate([2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018]):
        all_predictions.append(pd.read_pickle("results/" + name+"_"+str(int(year))+"_"+algo+".pkl"))
    all_var = pd.concat(all_predictions)
    df_all_predictions[algo] = all_var.iloc[:,0]
    df_all_predictions.index = all_var.index    
df_all_predictions.to_pickle("results/" + "df_all_"+name+".pkl")

for idx_year, year in enumerate([2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018]):
    for algo in ["lasso", "poly", "rf", "dnn", "transformer", "mar"]:
        os.remove("results/" + name+"_"+str(int(year))+"_"+algo+".pkl")


import seaborn as sns
observations = D17_total["FC_2_y"][D17_total.index.isin(df_all_predictions.index)]

observation_flux = observations[observations>=0]
df_all = df_all_predictions[df_all_predictions.index.isin(observation_flux.index)]
observations = observation_flux.values

plt.figure()
sns.boxplot(data=df_all.apply(lambda x: np.abs(x-observations)), showfliers=False, showmeans=True,
            meanprops={"marker":"d",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"white",
                      "markersize":"2"})
plt.title("Mean absolute error")

plt.figure()
sns.boxplot(data=df_all.apply(lambda x: x-observations), showfliers=False, showmeans=True,
            meanprops={"marker":"d",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"white",
                      "markersize":"2"})
plt.title("Bias")

observations = D17_total["FC_2_y"][D17_total.index.isin(df_all_predictions.index)]
plt.figure()
sns.barplot(data=dataframe_to_POD(df_all_predictions, observations))
plt.ylim((60, 100))
plt.title("Probability of detection")
plt.figure()
sns.barplot(data=dataframe_to_FAR(df_all_predictions, observations))
plt.title("False alarm ratio")
"""