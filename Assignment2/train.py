import pandas as pd
import numpy as np
import pickle
import pickle_compat
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC

pickle_compat.patch()


def calcAbsMean(parameter):
    abs_mean = 0
    for param in range(0, len(parameter) - 1):
        abs_mean = abs_mean + np.abs(parameter[(param + 1)] - parameter[param])
    return abs_mean / len(parameter)


def calcGCEntropy(parameter):
    len_param = len(parameter)
    entropy = 0
    if len_param <= 1:
        return 0
    else:
        value, ctr = np.unique(parameter, return_counts=True)
        ratio = ctr / len_param
        ratio_nonzero = np.count_nonzero(ratio)
        if ratio_nonzero <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy


def calcRMS(parameter):
    rms = 0
    for param in range(0, len(parameter) - 1):

        rms = rms + np.square(parameter[param])
    return np.sqrt(rms / len(parameter))


def calcFFT(parameter):
    ffourier = fft(parameter)
    param_len = len(parameter)

    t = 2 / 300
    ampl = []
    freq = np.linspace(0, param_len * t, param_len)

    for amp in ffourier:
        ampl.append(np.abs(amp))

    sorted_amp = ampl
    sorted_amp = sorted(sorted_amp)
    max_amp = sorted_amp[(-2)]
    max_freq = freq.tolist()[ampl.index(max_amp)]
    return [max_amp, max_freq]


def getNoMealTimes(time, timeDiff):
    times_arr = []
    time1 = time[0 : len(time) - 1]
    time2 = time[1 : len(time)]
    diff = list(np.array(time1) - np.array(time2))
    diff_list = list(zip(time1, time2, diff))
    for dl in diff_list:
        if dl[2] < timeDiff:
            times_arr.append(dl[0])
    return times_arr


def getNoMealData(mealTimes, startTime, endTime, isMealData, gcData):
    mealData_list = []

    for mTime in mealTimes:
        startIndex_meal = gcData[
            gcData["datetime"].between(
                mTime + pd.DateOffset(hours=startTime),
                mTime + pd.DateOffset(hours=endTime),
            )
        ]

        if startIndex_meal.shape[0] < 24:
            continue

        gc_values = startIndex_meal["Sensor Glucose (mg/dL)"].to_numpy()
        mean = startIndex_meal["Sensor Glucose (mg/dL)"].mean()

        if isMealData:
            missingGC_ctr = 30 - len(gc_values)
            if missingGC_ctr > 0:
                for i in range(missingGC_ctr):
                    gc_values = np.append(gc_values, mean)
            mealData_list.append(gc_values[0:30])
        else:
            mealData_list.append(gc_values[0:24])

    return pd.DataFrame(data=mealData_list)


def calcGCFeatures(data_meal_nomeal):
    gc_features = pd.DataFrame()
    for i in range(0, data_meal_nomeal.shape[0]):
        param = data_meal_nomeal.iloc[i, :].tolist()
        gc_features = gc_features.append(
            {
                "Minimum Value": min(param),
                "Maximum Value": max(param),
                "Mean of Absolute Values1": calcAbsMean(param[:13]),
                "Mean of Absolute Values2": calcAbsMean(param[13:]),
                "Root Mean Square": calcRMS(param),
                "Entropy": calcGCEntropy(param),
                "Max FFT Amplitude1": calcFFT(param[:13])[0],
                "Max FFT Frequency1": calcFFT(param[:13])[1],
                "Max FFT Amplitude2": calcFFT(param[13:])[0],
                "Max FFT Frequency2": calcFFT(param[13:])[1],
            },
            ignore_index=True,
        )
    return gc_features


def processData(insulinData, gcData):
    meal_data = pd.DataFrame()
    noMeal_data = pd.DataFrame()
    insulinData = insulinData[::-1]
    gcData = gcData[::-1]
    gcData["Sensor Glucose (mg/dL)"] = gcData["Sensor Glucose (mg/dL)"].interpolate(
        method="linear", limit_direction="both"
    )

    insulinData["datetime"] = pd.to_datetime(
        insulinData["Date"].astype(str) + " " + insulinData["Time"].astype(str)
    )
    gcData["datetime"] = pd.to_datetime(
        gcData["Date"].astype(str) + " " + gcData["Time"].astype(str)
    )

    processed_insulinData = insulinData[["datetime", "BWZ Carb Input (grams)"]]
    processed_gcData = gcData[["datetime", "Sensor Glucose (mg/dL)"]]

    processed_insulinData = processed_insulinData[
        (processed_insulinData["BWZ Carb Input (grams)"].notna())
        & (processed_insulinData["BWZ Carb Input (grams)"] > 0)
    ]

    times_list = list(processed_insulinData["datetime"])

    meal_times = []
    noMeal_times = []
    meal_times = getNoMealTimes(times_list, pd.Timedelta("0 days 120 min"))
    noMeal_times = getNoMealTimes(times_list, pd.Timedelta("0 days 240 min"))

    meal_data = getNoMealData(meal_times, -0.5, 2, True, processed_gcData)
    noMeal_data = getNoMealData(noMeal_times, 2, 4, False, processed_gcData)
    features_meal_data = calcGCFeatures(meal_data)
    features_noMeal_data = calcGCFeatures(noMeal_data)

    ss = StandardScaler()
    ss_meal = ss.fit_transform(features_meal_data)
    ss_noMeal = ss.fit_transform(features_noMeal_data)

    pca = PCA(n_components=5)
    pca.fit(ss_meal)

    pca_meal = pd.DataFrame(pca.fit_transform(ss_meal))
    pca_noMeal = pd.DataFrame(pca.fit_transform(ss_noMeal))

    pca_meal["class"] = 1
    pca_noMeal["class"] = 0

    data = pca_meal.append(pca_noMeal)
    data.index = [i for i in range(data.shape[0])]
    return data


if __name__ == "__main__":
    Insulin_patient2 = pd.read_csv("Insulin_patient2.csv")
    CGM_patient2 = pd.read_csv("CGM_patient2.csv")
    InsulinData = pd.read_csv("InsulinData.csv", low_memory=False)
    CGMData = pd.read_csv("CGMData.csv", low_memory=False)
    insulin_data = pd.concat([Insulin_patient2, InsulinData])
    gc_data = pd.concat([CGM_patient2, CGMData])

    data = processData(insulin_data, gc_data)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    model = SVC(kernel="linear", C=1, gamma=0.1)
    k_fold = KFold(5, True, 1)
    for train, test in k_fold.split(X, Y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        Y_train, Y_test = Y.iloc[train], Y.iloc[test]

        model.fit(X_train, Y_train)

    with open("Model.pkl", "wb") as (file):
        pickle.dump(model, file)
