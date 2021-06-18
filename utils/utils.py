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
    for algo in ["lasso", "poly", "rf", "dnn", "mar", "ensemble"]:
        print(algo, np.nanmean(results[algo]["rmse"]))

    print("\ncorr_coeff")
    for algo in ["lasso", "poly", "rf", "dnn", "mar", "ensemble"]:
        print(algo, np.nanmean(results[algo]["corr_coeff"]))

    print("\nbias")
    for algo in ["lasso", "poly", "rf", "dnn", "mar", "ensemble"]:
        print(algo, np.nanmean(results[algo]["bias"]))