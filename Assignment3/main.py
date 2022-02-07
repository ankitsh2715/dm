import pandas as pd
import numpy as np
import pickle
import math

from scipy import stats
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import pickle_compat
pickle_compat.patch()

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

def rootMeanSquare(val):
    rms = 0
    for p in range(0, len(val) - 1):
        rms = rms + np.square(val[p])

    return np.sqrt(rms / len(val))


def fastFourierTransform(val):
    ffourier = fft(val)
    val_len = len(val)
    t = 2 / 300
    freq = []
    frequency = np.linspace(0, val_len * t, val_len)
    for amp in ffourier:
        freq.append(np.abs(amp))
    sorted_amp = freq
    sorted_amp = sorted(sorted_amp)
    max_amp = sorted_amp[(-2)]
    max_freq = frequency.tolist()[freq.index(max_amp)]
    return [max_amp, max_freq]


def zeroCrossings(row, xAxis):
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


def extractGlucoseFeatures(meal_nomeal_data):
    gc_features = pd.DataFrame()
    
    for i in range(0, meal_nomeal_data.shape[0]):
        meal_nomeal_list = meal_nomeal_data.iloc[i, :].tolist()
        gc_features = gc_features.append(
            {
                "Minimum Value": min(meal_nomeal_list),
                "Maximum Value": max(meal_nomeal_list),
                "Mean of Absolute Values1": absoluteValueMean(meal_nomeal_list[:13]),
                "Mean of Absolute Values2": absoluteValueMean(meal_nomeal_list[13:]),
                "Max_Zero_Crossing": zeroCrossings(
                    meal_nomeal_list, meal_nomeal_data.shape[1]
                )[0],
                "Zero_Crossing_Rate": zeroCrossings(
                    meal_nomeal_list, meal_nomeal_data.shape[1]
                )[1],
                "Root Mean Square": rootMeanSquare(meal_nomeal_list),
                "Entropy": rootMeanSquare(meal_nomeal_list),
                "Max FFT Amplitude1": fastFourierTransform(meal_nomeal_list[:13])[0],
                "Max FFT Frequency1": fastFourierTransform(meal_nomeal_list[:13])[1],
                "Max FFT Amplitude2": fastFourierTransform(meal_nomeal_list[13:])[0],
                "Max FFT Frequency2": fastFourierTransform(meal_nomeal_list[13:])[1],
            },
            ignore_index=True,
        )
    return gc_features


def extractFeatures(meal_data):
    meal_gc_features = extractGlucoseFeatures(meal_data.iloc[:, :-1])
    sscaler = StandardScaler()
    meal_std = sscaler.fit_transform(meal_gc_features)
    pca = PCA(n_components=12)
    pca.fit(meal_std)
    with open("pcs_glucose_data.pkl", "wb") as (file):
        pickle.dump(pca, file)

    pca_meal = pd.DataFrame(pca.fit_transform(meal_std))

    return pca_meal


def calculateEntropy(bins):
    meal_entropy = []

    for bin_val in bins:
        bin_insulin = np.array(bin_val)
        bin_insulin = bin_insulin / float(bin_insulin.sum())
        bin_entropy = (
            bin_insulin
            * [np.log2(glucose) if glucose != 0 else 0 for glucose in bin_insulin]
        ).sum()
        meal_entropy += [bin_entropy]

    return meal_entropy


def calculatePurity(bins):
    purity = []

    for bin_val in bins:
        bin_insulin = np.array(bin_val)
        bin_insulin = bin_insulin / float(bin_insulin.sum())
        bin_purity = bin_insulin.max()
        purity += [bin_purity]

    return purity


def calculateDBScanSSE(dbscan_sse, centroid, meal_pca):
    for i in centroid.index:
        dbscan_sse = 0
        for index, row in meal_pca[meal_pca["clusters"] == i].iterrows():
            centroid_list = list(centroid.iloc[0, :])
            meal_list = list(row[:-1])
            for j in range(0, 12):
                dbscan_sse += (centroid_list[j] - meal_list[j]) ** 2

    return dbscan_sse


def groundTruthClusterMatrix(ground_truth, clusters, k_val):
    clusterMatrix = np.zeros((k_val, k_val))

    for i, j in enumerate(ground_truth):
        temp1 = j
        temp2 = clusters[i]
        clusterMatrix[temp1, temp2] += 1

    return clusterMatrix


if __name__ == "__main__":

    insulin_data = pd.read_csv("InsulinData.csv", low_memory=False)
    glucose_data = pd.read_csv("CGMData.csv", low_memory=False)
    bins = 7
    patient_data, insulin_lvls = processMealData(insulin_data, glucose_data)
    pca_meal = extractFeatures(patient_data)
    kmeans = KMeans(n_clusters=bins, max_iter=15000)
    kmeans.fit_predict(pca_meal)
    pLabels = list(kmeans.labels_)

    df = pd.DataFrame()
    df["bins"] = insulin_lvls
    df["kmeans_clusters"] = pLabels
    matrix_cluster = groundTruthClusterMatrix(df["bins"], df["kmeans_clusters"], 7)
    entropy_cluster = calculateEntropy(matrix_cluster)
    purity_cluster = calculatePurity(matrix_cluster)

    total = np.array([insulinBin.sum() for insulinBin in matrix_cluster])
    num_bins = total / float(total.sum())

    kmeans_sse = kmeans.inertia_
    entropy_kmeans = -(entropy_cluster * num_bins).sum()
    purity_kmeans = (purity_cluster * num_bins).sum()

    df_dbscan = pd.DataFrame()
    db_scan = DBSCAN(eps=0.127, min_samples=bins)
    clusters = db_scan.fit_predict(pca_meal)
    df_dbscan = pd.DataFrame(
        {
            "pc1": list(pca_meal.iloc[:, 0]),
            "pc2": list(pca_meal.iloc[:, 1]),
            "clusters": list(clusters),
        }
    )
    df_outliers = df_dbscan[df_dbscan["clusters"] == -1].iloc[:, 0:2]

    max_val_dbscan = max(df_dbscan["clusters"])
    while max_val_dbscan < bins - 1:
        largest_cluster = stats.mode(df_dbscan["clusters"]).mode[0]
        df_bi_cluster = df_dbscan[
            df_dbscan["clusters"] == stats.mode(df_dbscan["clusters"]).mode[0]
        ]
        bi_kmeans = KMeans(n_clusters=2, max_iter=1000, algorithm="auto").fit(
            df_bi_cluster
        )
        bi_pLabels = list(bi_kmeans.labels_)
        df_bi_cluster["bi_pcluster"] = bi_pLabels
        df_bi_cluster = df_bi_cluster.replace(to_replace=0, value=largest_cluster)
        df_bi_cluster = df_bi_cluster.replace(
            to_replace=1, value=max(df_dbscan["clusters"]) + 1
        )
        for x, y in zip(df_bi_cluster["pc1"], df_bi_cluster["pc2"]):
            newLabel = df_bi_cluster.loc[
                (df_bi_cluster["pc1"] == x) & (df_bi_cluster["pc2"] == y)
            ]
            df_dbscan.loc[
                (df_dbscan["pc1"] == x) & (df_dbscan["pc2"] == y), "clusters"
            ] = newLabel["bi_pcluster"]
        df["clusters"] = df_dbscan["clusters"]
        max_val_dbscan += 1

    dbscan_cluster_matrix = groundTruthClusterMatrix(
        df["bins"], df_dbscan["clusters"], 7
    )
    entropy_db_cluster = calculateEntropy(dbscan_cluster_matrix)
    total = np.array([insulinBin.sum() for insulinBin in dbscan_cluster_matrix])
    num_bins = total / float(total.sum())
    entropy_dbscan = -(entropy_db_cluster * num_bins).sum()
    purity_db_cluster = calculatePurity(dbscan_cluster_matrix)
    meal_pca2 = pca_meal.join(df_dbscan["clusters"])
    centroids = meal_pca2.groupby(df_dbscan["clusters"]).mean()
    sse_dbscan = calculateDBScanSSE(0, centroids.iloc[:, :12], meal_pca2)
    purity_dbscan = (purity_db_cluster * num_bins).sum()
    

    df_output = pd.DataFrame(
        [
            [
                kmeans_sse,
                sse_dbscan,
                entropy_kmeans,
                entropy_dbscan,
                purity_kmeans,
                purity_dbscan,
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
    df_output = df_output.fillna(0)
    df_output.to_csv("Results.csv", index=False, header=None)
