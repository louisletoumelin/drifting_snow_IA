import matplotlib.pyplot as plt
import seaborn as sns
from metrics.confusion_matrix import *


def plot_all_result(D17_total, name="results_2"):

    df_all_predictions = pd.read_pickle("results/" + "df_all_"+name+".pkl")
    observations = D17_total["FC_2_y"][D17_total.index.isin(df_all_predictions.index)]

    observation_flux = observations[observations >= 0]
    df_all = df_all_predictions[df_all_predictions.index.isin(observation_flux.index)]
    observations = observation_flux.values

    plt.figure()
    sns.boxplot(data=df_all.apply(lambda x: np.abs(x-observations)), showfliers=False, showmeans=True,
                meanprops={"marker": "d",
                           "markerfacecolor": "white",
                           "markeredgecolor": "white",
                          "markersize": "2"})
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

    df_training = pd.DataFrame()
    for name_column, name in zip(["Wind, Humidity, Temperature", "Wind, Hum., Temp. \n + obs. at higher elevation", "Wind, Hum., Temp. \n + precomputed vertical gradient", "Wind only", "Wind, Hum., Temp. \n + snowfalls from MAR"],["results_1", "results_2", "results_3", "results_5", "results_6"]):
        df_all_predictions = pd.read_pickle("results/" + "df_all_" + name + ".pkl")
        df_training[name_column] = df_all_predictions["transformer"]

    sns.boxplot(data=df_training.apply(lambda x: np.abs(x-observations)), showfliers=False, showmeans=True,
                meanprops={"marker":"d",
                           "markerfacecolor":"white",
                           "markeredgecolor":"white",
                          "markersize":"2"})
    plt.title("Mean absolute error for transformer architecture")


"""
['U2m', 'RH2m', 'T2m', 'RH3', 'RH4', 'RH5', 'RH6', 'T2', 'T3', 'T4', 'T5', 'FC_2_y']
rmse
lasso 40.85886531097823
poly 37.06573644190367
rf 39.93645169762181
dnn 37.74360046307183
transformer 36.79303755206936
mar 51.52221583138501

corr_coeff
lasso 0.7465435352755561
poly 0.8035682647251203
rf 0.7997321031497086
dnn 0.7866979711881441
transformer 0.8067338028949855
mar 0.5166040251741266

bias
lasso 10.914144008699143
poly 6.859834986362199
rf 7.983859807855506
dnn 4.791581306123437
transformer 4.175320138237399

 ['RH2m', 'T2m', 'U2m', 'gradient_zT1', 'gradient_zT2', 'gradient_zT3', 'gradient_zT4', 'gradient_zT5', 'gradient_zT6', 'FC_2_y']
rmse
lasso 40.938821906712306
poly 37.47432371357646
rf 40.20251237956741
dnn 39.18228643145219
transformer 36.98198059990684
mar 51.52872621903281

corr_coeff
lasso 0.7463742939892053
poly 0.8011709178681966
rf 0.7805343548266349
dnn 0.7864900728223845
transformer 0.7972763655676103
mar 0.5164520091326796

bias
lasso 10.990807329178601
poly 7.078663449497594
rf 6.3729868548735045
dnn 5.0689385579700845
transformer 5.64960936852624


['U2m', 'FC_2_y']
rmse
lasso 41.54421522699869
poly 38.75304784042551
rf 39.055083576526904
dnn 38.621023983426326
transformer 39.34134815290485
mar 51.532072170535606

corr_coeff
lasso 0.7344684633967593
poly 0.7775096680638214
rf 0.7764090877469351
dnn 0.7714746296623074
transformer 0.7767809898620773
mar 0.5164410198590934

bias
  print(algo, np.nanmean(results[algo]["corr_coeff"]))
lasso 10.598891623770744
poly 6.349799358726206
rf 5.854009168208567
dnn 6.001318231488695
transformer 5.382130804710858

['U2m', 'RH2m', 'T2m', 'SF', 'FC_2_y']
rmse
lasso 40.979769913570614
poly 37.33330614246083
rf 40.31356914387216
dnn 39.03545187801963
transformer 37.19297366717197
mar 51.532072170535606

corr_coeff
lasso 0.7459046939862235
poly 0.8026835079737222
rf 0.782992875254175
dnn 0.7889754529088064
transformer 0.8014258856699511
mar 0.5164410198590934

bias
lasso 10.995787100185556
poly 7.026873946949841
rf 6.67049179208024
dnn 5.480142761604604
transformer 3.387033942593626
"""
