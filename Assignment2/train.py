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
    gcData["Sensor Glucose (mg/dL)"] = gcData[
        "Sensor Glucose (mg/dL)"
    ].interpolate(method="linear", limit_direction="both")

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
    features_meal_data = glucoseFeatures(meal_data)
    features_noMeal_data = glucoseFeatures(noMeal_data)

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


def fn_zero_crossings(row, xAxis):
    slopes = [0]
    zero_cross = list()
    zero_crossing_rate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            zero_cross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    zero_crossing_rate = np.sum(
        [
            np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i]))
            for i in range(0, len(slopes) - 1)
        ]
    ) / (2 * len(slopes))
    if len(zero_cross) > 0:
        return [max(zero_cross)[0], zero_crossing_rate]
    else:
        return [0, 0]


def absoluteValueMean(param):
    meanValue = 0
    for p in range(0, len(param) - 1):
        meanValue = meanValue + np.abs(param[(p + 1)] - param[p])
    return meanValue / len(param)


def glucoseEntropy(param):
    paramLen = len(param)
    entropy = 0
    if paramLen <= 1:
        return 0
    else:
        value, count = np.unique(param, return_counts=True)
        ratio = count / paramLen
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy


def rootMeanSquare(param):
    rootMeanSquare = 0
    for p in range(0, len(param) - 1):

        rootMeanSquare = rootMeanSquare + np.square(param[p])
    return np.sqrt(rootMeanSquare / len(param))


def fastFourier(param):
    fastFourier = fft(param)
    paramLen = len(param)
    t = 2 / 300
    amplitude = []
    frequency = np.linspace(0, paramLen * t, paramLen)
    for amp in fastFourier:
        amplitude.append(np.abs(amp))
    sortedAmplitude = amplitude
    sortedAmplitude = sorted(sortedAmplitude)
    max_amplitude = sortedAmplitude[(-2)]
    max_frequency = frequency.tolist()[amplitude.index(max_amplitude)]
    return [max_amplitude, max_frequency]


def glucoseFeatures(meal_Nomeal_data):
    glucoseFeatures = pd.DataFrame()
    for i in range(0, meal_Nomeal_data.shape[0]):
        param = meal_Nomeal_data.iloc[i, :].tolist()
        glucoseFeatures = glucoseFeatures.append(
            {
                "Minimum Value": min(param),
                "Maximum Value": max(param),
                "Mean of Absolute Values1": absoluteValueMean(param[:13]),
                "Mean of Absolute Values2": absoluteValueMean(param[13:]),
                "Root Mean Square": rootMeanSquare(param),
                "Entropy": rootMeanSquare(param),
                "Max FFT Amplitude1": fastFourier(param[:13])[0],
                "Max FFT Frequency1": fastFourier(param[:13])[1],
                "Max FFT Amplitude2": fastFourier(param[13:])[0],
                "Max FFT Frequency2": fastFourier(param[13:])[1],
            },
            ignore_index=True,
        )
    return glucoseFeatures


if __name__ == "__main__":
    insulin_data_1 = pd.read_csv("Insulin_patient2.csv")
    glucose_data_1 = pd.read_csv("CGM_patient2.csv")
    insulin_data_2 = pd.read_csv("InsulinData.csv", low_memory=False)
    glucose_data_2 = pd.read_csv("CGMData.csv", low_memory=False)
    insulin_data = pd.concat([insulin_data_1, insulin_data_2])
    glucose_data = pd.concat([glucose_data_1, glucose_data_2])
    data = processData(insulin_data, glucose_data)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    model = SVC(kernel="linear", C=1, gamma=0.1)
    kfold = KFold(5, True, 1)
    for tr, tst in kfold.split(X, Y):
        X_train, X_test = X.iloc[tr], X.iloc[tst]
        Y_train, Y_test = Y.iloc[tr], Y.iloc[tst]

        model.fit(X_train, Y_train)

    with open("Model.pkl", "wb") as (file):
        pickle.dump(model, file)
