import numpy as np
import pandas as pd


def dataframe_to_POD(df_pred, df_obs):
    """
    Convert fluxes to detection of drifting-snow events.

    A drifting snow event occurs if drifting-snow mass flux > 1 g/m²/s.
    """
    observed = df_obs >= 1
    # not_observed = df_obs[df_obs < 1]

    results = pd.DataFrame()
    for column in df_pred.columns:
        simulated = df_pred[column] >= 1
        not_simulated = df_pred[column] < 1

        obs_and_sim = df_pred[column][observed & simulated].count()
        obs_and_not_sim = df_pred[column][observed & not_simulated].count()
        results[column] = [POD(obs_and_sim, obs_and_not_sim)]

    return results


def dataframe_to_FAR(df_pred, df_obs):
    """
    Convert fluxes to detection of drifting-snow events.

    A drifting snow event occurs if drifting-snow mass flux > 1 g/m²/s.
    """
    observed = df_obs >= 1
    # not_observed = df_obs[df_obs < 1]

    results = pd.DataFrame()
    for column in df_pred.columns:
        simulated = df_pred[column] >= 1
        not_simulated = df_pred[column] < 1

        obs_and_sim = df_pred[column][observed & simulated].count()
        obs_and_not_sim = df_pred[column][observed & not_simulated].count()
        results[column] = [FAR(obs_and_sim, obs_and_not_sim)]

    return results


def POD(obs_and_sim, obs_and_not_sim):
    """Probability of detection"""
    good_sim = obs_and_sim
    all_sim = obs_and_sim + obs_and_not_sim
    return 100 * np.round(good_sim / all_sim, 3)


def FAR(obs_and_sim, not_obs_and_sim):
    """False alarm ratio"""
    false_detection = not_obs_and_sim
    all_sim = not_obs_and_sim + obs_and_sim
    return 100 * np.round(false_detection / all_sim, 3)
