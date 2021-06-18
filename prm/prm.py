def PRM():
    """
    Define parameters
    """

    prm = {

        # 1 = Classic, 2= Concatenate, 3 = BatchNormalization
        "architecture": 1,

        # For architecture 1 and 2
        "n_layers": 4,
        "n_neurons": 200,
        "optimizer": "Nadam",
        'kernel_initializer': "he_uniform",
        'activation': "tanh",
        "learning_rate": 0.01,
        'batch_size': 16,
        'epochs': 50,
        'dropout': 0,
        "loss": "mean_squared_error",
        "GPU": False}

    prm = check_path_data(prm)
    print(prm["path_data"])
    return prm


def check_path_data(prm):
    """
    Define data path depending on which machine we work on.
    """

    if prm["GPU"]:
        prm["path_data"] = "//scratch/mrmn/letoumelinl/FC_predict/"
    else:
        prm["path_data"] = "data/"
    return prm
