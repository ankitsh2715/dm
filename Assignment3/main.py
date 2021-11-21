import pandas as pd
import numpy as np
import pickle
import math
import pickle_compat
pickle_compat.patch()

from scipy import stats
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def processMealDataHelper(meal_time, start_time, end_time, insulin_level, processed_gluc_data):
    new_meal_data = []

    for j, mTime in enumerate(meal_time):

        meal_start_index = processed_gluc_data[
            processed_gluc_data["datetime"].between(
                mTime + pd.DateOffset(hours=start_time),
                mTime + pd.DateOffset(hours=end_time),
            )
        ]

        if meal_start_index.shape[0] < 8:
            del insulin_level[j]
            continue
        
        gluc_values = meal_start_index["Sensor Glucose (mg/dL)"].to_numpy()
        mean_value = meal_start_index["Sensor Glucose (mg/dL)"].mean()
        missing_values_count = 30 - len(gluc_values)

        if missing_values_count > 0:
            for i in range(missing_values_count):
                gluc_values = np.append(gluc_values, mean_value)
        new_meal_data.append(gluc_values[0:30])

    return pd.DataFrame(data=new_meal_data), insulin_level


def getMealtimes(insulin_data):
    filter_time = []
    insulin_val = []
    insulin_lvl = []
    filter_time_1 = []
    filter_time_2 = []
    meal_time_res = []
    diff = []

    carb_input = insulin_data["BWZ Carb Input (grams)"]
    max_val = carb_input.max()
    min_val = carb_input.min()
    bins = math.ceil(max_val - min_val / 60)

    for i in insulin_data["datetime"]:
        filter_time.append(i)

    for i in insulin_data["BWZ Carb Input (grams)"]:
        insulin_val.append(i)
    
    for i, j in enumerate(filter_time):
        if i < len(filter_time) - 1:
            diff.append((filter_time[i + 1] - filter_time[i]).total_seconds() / 3600)
    
    filter_time_1 = filter_time[0:-1]
    filter_time_2 = filter_time[1:]
    bins = []
    for i in insulin_val[0:-1]:
        bins.append(
            0
            if (i >= min_val and i <= min_val + 20)
            else 1
            if (i >= min_val + 21 and i <= min_val + 40)
            else 2
            if (i >= min_val + 41 and i <= min_val + 60)
            else 3
            if (i >= min_val + 61 and i <= min_val + 80)
            else 4
            if (i >= min_val + 81 and i <= min_val + 100)
            else 5
            if (i >= min_val + 101 and i <= min_val + 120)
            else 6
        )
    reqValues = list(zip(filter_time_1, filter_time_2, diff, bins))
    for j in reqValues:
        if j[2] > 2.5:
            meal_time_res.append(j[0])
            insulin_lvl.append(j[3])
        else:
            continue
    return meal_time_res, insulin_lvl

def processMealData(insulin_data, glucose_data):
    meal_data = pd.DataFrame()
    glucose_data["Sensor Glucose (mg/dL)"] = glucose_data[
        "Sensor Glucose (mg/dL)"
    ].interpolate(method="linear", limit_direction="both")
    insulin_data = insulin_data[::-1]
    glucose_data = glucose_data[::-1]

    insulin_data["datetime"] = insulin_data["Date"] + " " + insulin_data["Time"]
    insulin_data["datetime"] = pd.to_datetime(insulin_data["datetime"])
    glucose_data["datetime"] = glucose_data["Date"] + " " + glucose_data["Time"]
    glucose_data["datetime"] = pd.to_datetime(insulin_data["datetime"])

    insulin_data_filter = insulin_data[["datetime", "BWZ Carb Input (grams)"]]
    insulin_data_final = insulin_data_filter[
        (insulin_data_filter["BWZ Carb Input (grams)"] > 0)
    ]
    gc_data_filter = glucose_data[["datetime", "Sensor Glucose (mg/dL)"]]

    mealTimes, insulinLevels = getMealtimes(insulin_data_final)
    meal_data, new_insulinLevels = processMealDataHelper(
        mealTimes, -0.5, 2, insulinLevels, gc_data_filter
    )

    return meal_data, new_insulinLevels


def absoluteValueMean(val):
    abs_mean = 0
    for p in range(0, len(val) - 1):
        abs_mean = abs_mean + np.abs(val[(p + 1)] - val[p])
    return abs_mean / len(val)


def glucoseEntropy(val):
    entropy = 0
    val_size = len(val)
    
    if val_size <= 1:
        return 0
    else:
        value, count = np.unique(val, return_counts=True)
        ratio = count / val_size
        non_zero_ratio = np.count_nonzero(ratio)
        
        if non_zero_ratio <= 1:
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
                "Max_Zero_Crossing": fn_zero_crossings(
                    param, meal_Nomeal_data.shape[1]
                )[0],
                "Zero_Crossing_Rate": fn_zero_crossings(
                    param, meal_Nomeal_data.shape[1]
                )[1],
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


def getFeatures(mealData):
    mealDataFeatures = glucoseFeatures(mealData.iloc[:, :-1])

    stdScaler = StandardScaler()
    meal_std = stdScaler.fit_transform(mealDataFeatures)

    pca = PCA(n_components=12)
    pca.fit(meal_std)

    with open("pcs_glucose_data.pkl", "wb") as (file):
        pickle.dump(pca, file)

    meal_pca = pd.DataFrame(pca.fit_transform(meal_std))
    return meal_pca


def compute_Entropy(bins):
    mealEntropy = []
    for insulinBin in bins:
        insulinBin = np.array(insulinBin)
        insulinBin = insulinBin / float(insulinBin.sum())
        binEntropy = (
            insulinBin
            * [np.log2(glucose) if glucose != 0 else 0 for glucose in insulinBin]
        ).sum()
        mealEntropy += [binEntropy]

    return mealEntropy


def compute_Purity(bins):
    mealPurity = []
    for insulinBin in bins:
        insulinBin = np.array(insulinBin)
        insulinBin = insulinBin / float(insulinBin.sum())
        binPurity = insulinBin.max()
        mealPurity += [binPurity]
    return mealPurity


def computeDBSCAN_SSE(dbscan_sse, test, meal_pca2):
    for i in test.index:
        dbscan_sse = 0
        for index, row in meal_pca2[meal_pca2["clusters"] == i].iterrows():
            test_row = list(test.iloc[0, :])
            meal_row = list(row[:-1])
            for j in range(0, 12):
                dbscan_sse += (test_row[j] - meal_row[j]) ** 2
    return dbscan_sse


def clusterMatrixwithGroundTruth(groundTruth, Clustered, k):
    clusterMatrix = np.zeros((k, k))
    for i, j in enumerate(groundTruth):
        val1 = j
        val2 = Clustered[i]
        clusterMatrix[val1, val2] += 1
    return clusterMatrix


if __name__ == "__main__":

    insulin_data = pd.read_csv("InsulinData.csv", low_memory=False)
    glucose_data = pd.read_csv("CGMData.csv", low_memory=False)

    patient_data, insulinLevels = processMealData(insulin_data, glucose_data)

    meal_pca = getFeatures(patient_data)

    kmeans = KMeans(n_clusters=7, max_iter=7000)
    kmeans.fit_predict(meal_pca)

    pLabels = list(kmeans.labels_)

    df = pd.DataFrame()
    df["bins"] = insulinLevels
    df["kmeans_clusters"] = pLabels

    clusterMatrix = clusterMatrixwithGroundTruth(df["bins"], df["kmeans_clusters"], 7)
    cluster_entropy = compute_Entropy(clusterMatrix)
    cluster_purity = compute_Purity(clusterMatrix)

    totalCount = np.array([insulinBin.sum() for insulinBin in clusterMatrix])
    binCount = totalCount / float(totalCount.sum())

    kmeans_SSE = kmeans.inertia_
    kmeans_purity = (cluster_purity * binCount).sum()
    kmeans_entropy = -(cluster_entropy * binCount).sum()

    dbscan_df = pd.DataFrame()
    db = DBSCAN(eps=0.127, min_samples=7)
    clusters = db.fit_predict(meal_pca)
    dbscan_df = pd.DataFrame(
        {
            "pc1": list(meal_pca.iloc[:, 0]),
            "pc2": list(meal_pca.iloc[:, 1]),
            "clusters": list(clusters),
        }
    )
    outliers_df = dbscan_df[dbscan_df["clusters"] == -1].iloc[:, 0:2]

    initial_value = 0
    bins = 7
    i = max(dbscan_df["clusters"])
    while i < bins - 1:
        largestClusterLabel = stats.mode(dbscan_df["clusters"]).mode[0]
        biCluster_df = dbscan_df[
            dbscan_df["clusters"] == stats.mode(dbscan_df["clusters"]).mode[0]
        ]
        bi_kmeans = KMeans(n_clusters=2, max_iter=1000, algorithm="auto").fit(
            biCluster_df
        )
        bi_pLabels = list(bi_kmeans.labels_)
        biCluster_df["bi_pcluster"] = bi_pLabels
        biCluster_df = biCluster_df.replace(to_replace=0, value=largestClusterLabel)
        biCluster_df = biCluster_df.replace(
            to_replace=1, value=max(dbscan_df["clusters"]) + 1
        )
        for x, y in zip(biCluster_df["pc1"], biCluster_df["pc2"]):
            newLabel = biCluster_df.loc[
                (biCluster_df["pc1"] == x) & (biCluster_df["pc2"] == y)
            ]
            dbscan_df.loc[
                (dbscan_df["pc1"] == x) & (dbscan_df["pc2"] == y), "clusters"
            ] = newLabel["bi_pcluster"]
        df["clusters"] = dbscan_df["clusters"]
        i += 1

    clusterMatrix_dbscan = clusterMatrixwithGroundTruth(
        df["bins"], dbscan_df["clusters"], 7
    )

    cluster_entropy_db = compute_Entropy(clusterMatrix_dbscan)
    cluster_purity_db = compute_Purity(clusterMatrix_dbscan)
    totalCount = np.array([insulinBin.sum() for insulinBin in clusterMatrix_dbscan])
    binCount = totalCount / float(totalCount.sum())

    meal_pca2 = meal_pca.join(dbscan_df["clusters"])
    centroids = meal_pca2.groupby(dbscan_df["clusters"]).mean()

    dbscan_sse = computeDBSCAN_SSE(initial_value, centroids.iloc[:, :12], meal_pca2)
    dbscan_purity = (cluster_purity_db * binCount).sum()
    dbscan_entropy = -(cluster_entropy_db * binCount).sum()

    outputdf = pd.DataFrame(
        [
            [
                kmeans_SSE,
                dbscan_sse,
                kmeans_entropy,
                dbscan_entropy,
                kmeans_purity,
                dbscan_purity,
            ]
        ],
        columns=[
            "K-Means SSE",
            "DBSCAN SSE",
            "K-Means entropy",
            "DBSCAN entropy",
            "K-Means purity",
            "DBSCAN purity",
        ],
    )
    outputdf = outputdf.fillna(0)
    outputdf.to_csv("Results.csv", index=False, header=None)
