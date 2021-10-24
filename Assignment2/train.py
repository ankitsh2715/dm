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


def calcAbsoluteMean(parameter):
    abs_mean = 0
    for param in range(0, len(parameter) - 1):
        abs_mean = abs_mean + np.abs(parameter[(param + 1)] - parameter[param])
    return abs_mean / len(parameter)


def calcGcEntropy(parameter):
    param_len = len(parameter)
    gc_entropy = 0
    if param_len <= 1:
        return 0
    else:
        value, ctr = np.unique(parameter, return_counts=True)
        ratio = ctr / param_len
        ratio_nonzero = np.count_nonzero(ratio)
        if ratio_nonzero <= 1:
            return 0
        for i in ratio:
            gc_entropy -= i * np.log2(i)
        return gc_entropy


def calcRMS(parameter):
    rms = 0
    for param in range(0, len(parameter) - 1):

        rms = rms + np.square(parameter[param])
    return np.sqrt(rms / len(parameter))


def calcFFT(parameter):
    ffourier = fft(parameter)
    parameter_len = len(parameter)

    t = 2 / 300
    amp = []
    freq = np.linspace(0, parameter_len * t, parameter_len)

    for amp in ffourier:
        amp.append(np.abs(amp))

    sorted_amp = amp
    sorted_amp = sorted(sorted_amp)
    max_amp = sorted_amp[(-2)]
    max_freq = freq.tolist()[amp.index(max_amp)]

    return [max_amp, max_freq]


def getGCFeatures(data_Meal_NoMeal):
    gc_features = pd.DataFrame()

    for i in range(0, data_Meal_NoMeal.shape[0]):
        param = data_Meal_NoMeal.iloc[i, :].tolist()
        gc_features = gc_features.append(
            {
                "Minimum Value": min(param),
                "Maximum Value": max(param),
                "Mean of Absolute Values1": calcAbsoluteMean(param[:13]),
                "Mean of Absolute Values2": calcAbsoluteMean(param[13:]),
                "Root Mean Square": calcRMS(param),
                "Entropy": calcGcEntropy(param),
                "Max FFT Amplitude1": calcFFT(param[:13])[0],
                "Max FFT Frequency1": calcFFT(param[:13])[1],
                "Max FFT Amplitude2": calcFFT(param[13:])[0],
                "Max FFT Frequency2": calcFFT(param[13:])[1],
            },
            ignore_index=True,
        )

    return gc_features


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
    features_meal_data = getGCFeatures(meal_data)
    features_noMeal_data = getGCFeatures(noMeal_data)

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
    insulin_patient2 = pd.read_csv("Insulin_patient2.csv")
    cgm_patient2 = pd.read_csv("CGM_patient2.csv")
    insulin_data = pd.read_csv("InsulinData.csv", low_memory=False)
    cgm_data = pd.read_csv("CGMData.csv", low_memory=False)
    insulin_data = pd.concat([insulin_patient2, insulin_data])
    glucose_data = pd.concat([cgm_patient2, cgm_data])

    data = processData(insulin_data, glucose_data)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    model = SVC(kernel="linear", C=1, gamma=0.1)
    k_fold = KFold(5, True, 1)

    for train, test in k_fold.split(X, Y):
        train_x, test_x = X.iloc[train], X.iloc[test]
        train_y, test_y = Y.iloc[train], Y.iloc[test]

        model.fit(train_x, train_y)

    with open("Model.pkl", "wb") as (file):
        pickle.dump(model, file)
