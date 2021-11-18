import numpy as np
import pandas as pd
import math
import datetime
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

insulin_data = pd.read_csv('InsulinData.csv', sep=',', low_memory=False)
insulin_data['dateTime'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])
insulin_data = insulin_data.sort_values(by='dateTime', ascending=True)

cgm_data = pd.read_csv('CGMData.csv', sep=',', low_memory=False)
cgm_data['dateTime'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
cgm_data = cgm_data.sort_values(by='dateTime', ascending=True)

insulin_data['New Index'] = range(0, 0 + len(insulin_data))
meal_window = insulin_data.loc[insulin_data['BWZ Carb Input (grams)'] > 0][
    ['New Index', 'Date', 'Time', 'BWZ Carb Input (grams)', 'dateTime']]
meal_window['diff'] = meal_window['dateTime'].diff(periods=1)
meal_window['shiftUp'] = meal_window['diff'].shift(-1)

meal_window = meal_window.loc[(meal_window['shiftUp'] > datetime.timedelta(minutes=120)) | (pd.isnull(meal_window['shiftUp']))]

meal_data_cgm = pd.DataFrame()
meal_data_cgm['New Index'] = ""

for i in range(len(meal_window)):
    meal_window_before = meal_window['dateTime'].iloc[i] - datetime.timedelta(minutes=30)
    meal_window_after = meal_window['dateTime'].iloc[i] + datetime.timedelta(minutes=120)
    meal_window_cgm_data = cgm_data.loc[(cgm_data['dateTime'] >= meal_window_before) & (cgm_data['dateTime'] < meal_window_after)]
    
    arr = []
    idx = 0
    idx = meal_window['New Index'].iloc[i]

    for j in range(len(meal_window_cgm_data)):
        arr.append(meal_window_cgm_data['Sensor Glucose (mg/dL)'].iloc[j])

    meal_data_cgm = meal_data_cgm.append(pd.Series(arr), ignore_index=True)
    meal_data_cgm.iloc[i, meal_data_cgm.columns.get_loc('New Index')] = idx

meal_data_cgm['New Index'] = meal_data_cgm['New Index'].astype(int)
meal_data_cgm_idx = pd.DataFrame()
meal_data_cgm_idx['New Index'] = meal_data_cgm['New Index']
meal_data_cgm = meal_data_cgm.drop(columns='New Index')

total_rows = meal_data_cgm.shape[0]
total_columns = meal_data_cgm.shape[1]

meal_data_cgm.dropna(axis=0, how='all', thresh=total_columns / 4, subset=None, inplace=True)
meal_data_cgm.dropna(axis=1, how='all', thresh=total_rows / 4, subset=None, inplace=True)
meal_data_cgm.interpolate(axis=0, method='linear', limit_direction='forward', inplace=True)
meal_data_cgm.bfill(axis=1, inplace=True)

no_meal_data_cgm_idx = meal_data_cgm.copy()
meal_data_cgm = pd.merge(meal_data_cgm, meal_data_cgm_idx, left_index=True, right_index=True)
meal_data_cgm['mean CGM data'] = no_meal_data_cgm_idx.mean(axis=1)
meal_data_cgm['max-start_over_start'] = no_meal_data_cgm_idx.max(axis=1) / no_meal_data_cgm_idx[0]

meal_qty = meal_window[['BWZ Carb Input (grams)', 'New Index']]
meal_qty = meal_qty.rename(columns={'BWZ Carb Input (grams)': 'Meal Amount'})
meal_qty_lbl = pd.DataFrame()
meal_qty_lbl['Bin Label'] = meal_qty.apply(lambda row: get_bin_lbl(row['Meal Amount']).astype(np.int64), axis=1)
meal_qty_lbl['New Index'] = meal_qty['New Index']

meal_data_qty = meal_data_cgm.merge(meal_qty_lbl, how='inner', on=['New Index'])

meal_carbs_intake_period = pd.DataFrame()
meal_carbs_intake_period = meal_window[['BWZ Carb Input (grams)', 'New Index']]
meal_data_qty = meal_data_qty.merge(meal_carbs_intake_period, how='inner', on=['New Index'])
meal_data_qty = meal_data_qty.drop(columns='New Index')

features_carbs = pd.DataFrame()
features_carbs = meal_data_qty[['BWZ Carb Input (grams)', 'mean CGM data']]

KMeans_val = features_carbs.copy()
KMeans_val = KMeans_val.values.astype('float32', copy=False)
KMeans_data = StandardScaler().fit(KMeans_val)
scaler_features = KMeans_data.transform(KMeans_val)

KMeans = range(1, 16)
sse = []
for k in KMeans:
    test_features = KMeans(n_clusters=k)
    test_features.fit(scaler_features)
    sse.append(test_features.inertia_)

result = KMeans(n_clusters=10)
y_val_predict = result.fit_predict(scaler_features)
sse_KMeans = result.inertia_

features_carbs['cluster'] = y_val_predict
features_carbs.head()
result.cluster_centers_

groundTruth_data = meal_data_qty["Bin Label"].tolist()
bins_clusters_df = pd.DataFrame({'ground_true_arr': groundTruth_data, 'kmeans_labels': list(y_val_predict)},
                                columns=['ground_true_arr', 'kmeans_labels'])

confusion_matrix = pd.pivot_table(bins_clusters_df, index='kmeans_labels', columns='ground_true_arr', aggfunc=len)
confusion_matrix.fillna(value=0, inplace=True)
confusion_matrix = confusion_matrix.reset_index()
confusion_matrix = confusion_matrix.drop(columns=['kmeans_labels'])

confusion_matrix_copy = confusion_matrix.copy()
confusion_matrix_copy['Total'] = confusion_matrix.sum(axis=1)
confusion_matrix_copy['Row_entropy'] = confusion_matrix.apply(lambda row: get_entropy(row), axis=1)
data_sum = confusion_matrix_copy['Total'].sum()
confusion_matrix_copy['entropy_prob'] = confusion_matrix_copy['Total'] / data_sum * confusion_matrix_copy[
    'Row_entropy']

entropy_kmeans = confusion_matrix_copy['entropy_prob'].sum()

confusion_matrix_copy['Max_val'] = confusion_matrix.max(axis=1)
KMeans_purity_data = confusion_matrix_copy['Max_val'].sum() / data_sum;

features_dbscan = features_carbs.copy()[['BWZ Carb Input (grams)', 'mean CGM data']]
features_dbscan_arr = features_dbscan.values.astype('float32', copy=False)

dbscan_data_scaler = StandardScaler().fit(features_dbscan_arr)
features_dbscan_arr = dbscan_data_scaler.transform(features_dbscan_arr)
model = DBSCAN(eps=0.19, min_samples=5).fit(features_dbscan_arr)
features_carbs['cluster'] = model.labels_
colors = model.labels_
colors_clusters = colors[colors != -1]
color_outliers = 'black'

dbscan_bins_clusters = pd.DataFrame({'ground_true_arr': groundTruth_data, 'dbscan_labels': list(model.labels_)},
                                       columns=['ground_true_arr', 'dbscan_labels'])

dbscan_confusion_matrix = pd.pivot_table(dbscan_bins_clusters, index='ground_true_arr', columns='dbscan_labels',
                                         aggfunc=len)
dbscan_confusion_matrix.fillna(value=0, inplace=True)
dbscan_confusion_matrix = dbscan_confusion_matrix.reset_index()
dbscan_confusion_matrix = dbscan_confusion_matrix.drop(columns=['ground_true_arr'])
dbscan_confusion_matrix = dbscan_confusion_matrix.drop(columns=[-1])
dbscan_confusion_matrix_copy = dbscan_confusion_matrix.copy()

dbscan_confusion_matrix_copy['Total'] = dbscan_confusion_matrix.sum(axis=1)
dbscan_confusion_matrix_copy['Row_entropy'] = dbscan_confusion_matrix.apply(lambda row: get_entropy_dbscan(row), axis=1)
data_sum = dbscan_confusion_matrix_copy['Total'].sum()
dbscan_confusion_matrix_copy['entropy_prob'] = dbscan_confusion_matrix_copy['Total'] / data_sum * \
                                               dbscan_confusion_matrix_copy['Row_entropy']
dbscan_entropy = dbscan_confusion_matrix_copy['entropy_prob'].sum()

dbscan_confusion_matrix_copy['Max_val'] = dbscan_confusion_matrix.max(axis=1)
dbscan_purity = dbscan_confusion_matrix_copy['Max_val'].sum() / data_sum;

features_carbs = features_carbs.loc[features_carbs['cluster'] != -1]

centroid_features_dbscan = features_carbs.copy()
carb_input_centroid = {}
cgm_mean_centroid = {}
squared_error = {}

for i in range(len(dbscan_confusion_matrix.columns)):
    cluster_group = features_carbs.loc[features_carbs['cluster'] == i]
    centroid_carb_input = cluster_group['BWZ Carb Input (grams)'].mean()
    centroid_cgm_mean = cluster_group['mean CGM data'].mean()
    carb_input_centroid[i] = centroid_carb_input
    cgm_mean_centroid[i] = centroid_cgm_mean

centroid_features_dbscan['centroid_carb_input'] = features_carbs.apply(
    lambda row: centroid_carb_input_calc(row), axis=1)
centroid_features_dbscan['centroid_cgm_mean'] = features_carbs.apply(
    lambda row: centroid_cgm_mean_calc(row), axis=1)
centroid_features_dbscan['centroid_difference'] = 0

for i in range(len(centroid_features_dbscan)):
    centroid_features_dbscan['centroid_difference'].iloc[i] = math.pow(
        centroid_features_dbscan['BWZ Carb Input (grams)'].iloc[i] -
        centroid_features_dbscan['centroid_carb_input'].iloc[i], 2) + math.pow(
        centroid_features_dbscan['mean CGM data'].iloc[i] -
        centroid_features_dbscan['centroid_cgm_mean'].iloc[i], 2)

for i in range(len(dbscan_confusion_matrix.columns)):
    squared_error[i] = centroid_features_dbscan.loc[centroid_features_dbscan['cluster'] == i][
        'centroid_difference'].sum()

DBSCAN_SSE = 0
for i in squared_error:
    DBSCAN_SSE = DBSCAN_SSE + squared_error[i];

KMeans_DBSCAN = [sse_KMeans, DBSCAN_SSE, entropy_kmeans, dbscan_entropy, KMeans_purity_data, dbscan_purity]
results_df = pd.DataFrame(KMeans_DBSCAN).T
results_df.to_csv('Results.csv', header=False, index=False)


# utility methods
def get_bin_lbl(x):
    if (x <= 23):
        return np.floor(0);
    elif (x <= 43):
        return np.floor(1);
    elif (x <= 63):
        return np.floor(2);
    elif (x <= 83):
        return np.floor(3);
    elif (x <= 103):
        return np.floor(4);
    else:
        return np.floor(5);

def get_entropy(row):
    sum = 0
    entropy = 0
    for i in range(len(confusion_matrix.columns)):
        sum = sum + row[i];
    for j in range(len(confusion_matrix.columns)):
        if (row[j] == 0):
            continue;
        entropy = entropy + row[j] / sum * math.log2(row[j] / sum)
    return -entropy

def get_entropy_dbscan(row):
    total = 0
    entropy = 0
    for i in range(len(dbscan_confusion_matrix.columns)):
        total = total + row[i];

    for j in range(len(dbscan_confusion_matrix.columns)):
        if (row[j] == 0):
            continue;
        entropy = entropy + row[j] / total * math.log2(row[j] / total)
    return -entropy

def centroid_carb_input_calc(row):
    return carb_input_centroid[row['cluster']]

def centroid_cgm_mean_calc(row):
    return cgm_mean_centroid[row['cluster']]
