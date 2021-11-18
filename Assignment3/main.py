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
meal_window

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
cgm_Mealdata_index = pd.DataFrame()
cgm_Mealdata_index['New Index'] = meal_data_cgm['New Index']
meal_data_cgm = meal_data_cgm.drop(columns='New Index')

#Interpolation
total_rows = meal_data_cgm.shape[0]
total_columns = meal_data_cgm.shape[1]
meal_data_cgm.dropna(axis=0, how='all', thresh=total_columns / 4, subset=None, inplace=True)
meal_data_cgm.dropna(axis=1, how='all', thresh=total_rows / 4, subset=None, inplace=True)
meal_data_cgm.interpolate(axis=0, method='linear', limit_direction='forward', inplace=True)
meal_data_cgm.bfill(axis=1, inplace=True)
cgm_NoMealdata_index = meal_data_cgm.copy()
cgm_mean = meal_data_cgm.copy()

meal_data_cgm = pd.merge(meal_data_cgm, cgm_Mealdata_index, left_index=True, right_index=True)
meal_data_cgm['mean CGM data'] = cgm_NoMealdata_index.mean(axis=1)
meal_data_cgm['max-start_over_start'] = cgm_NoMealdata_index.max(axis=1) / cgm_NoMealdata_index[
    0]

meal_Quantity = meal_window[['BWZ Carb Input (grams)', 'New Index']]
meal_Quantity = meal_Quantity.rename(columns={'BWZ Carb Input (grams)': 'Meal Amount'})
max_meal = meal_Quantity['Meal Amount'].max()
min_meal = meal_Quantity['Meal Amount'].min()

meal_quantity_label = pd.DataFrame()


def bin_label(x):
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


meal_quantity_label['Bin Label'] = meal_Quantity.apply(lambda row: bin_label(row['Meal Amount']).astype(np.int64),
                                                       axis=1)
meal_quantity_label['New Index'] = meal_Quantity['New Index']

meal_data_quantity = meal_data_cgm.merge(meal_quantity_label, how='inner', on=['New Index'])

meal_carbohydrates_intake_time = pd.DataFrame()
meal_carbohydrates_intake_time = meal_window[['BWZ Carb Input (grams)', 'New Index']]
meal_data_quantity = meal_data_quantity.merge(meal_carbohydrates_intake_time, how='inner', on=['New Index'])
meal_data_quantity = meal_data_quantity.drop(columns='New Index')

carb_feature_extraction = pd.DataFrame()
carb_feature_extraction = meal_data_quantity[['BWZ Carb Input (grams)', 'mean CGM data']]

kmeans_value = carb_feature_extraction.copy()
kmeans_value = kmeans_value.values.astype('float32', copy=False)
kmeans_data = StandardScaler().fit(kmeans_value)
Feature_extraction_scaler = kmeans_data.transform(kmeans_value)

kmeans_range = range(1, 16)
sse = []
for k in kmeans_range:
    kmeans_feature_test = KMeans(n_clusters=k)
    kmeans_feature_test.fit(Feature_extraction_scaler)
    sse.append(kmeans_feature_test.inertia_)

kmeans_result = KMeans(n_clusters=10)
kmeans_predictionvalue_y = kmeans_result.fit_predict(Feature_extraction_scaler)
KMeans_sse = kmeans_result.inertia_

carb_feature_extraction['cluster'] = kmeans_predictionvalue_y
carb_feature_extraction.head()

kmeans_result.cluster_centers_

ground_truthdata_array = meal_data_quantity["Bin Label"].tolist()

bins_clusters_df = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'kmeans_labels': list(kmeans_predictionvalue_y)},
                                columns=['ground_true_arr', 'kmeans_labels'])

confusion_matrix_data = pd.pivot_table(bins_clusters_df, index='kmeans_labels', columns='ground_true_arr', aggfunc=len)
confusion_matrix_data.fillna(value=0, inplace=True)

confusion_matrix_data = confusion_matrix_data.reset_index()
confusion_matrix_data = confusion_matrix_data.drop(columns=['kmeans_labels'])

confusion_matrix_copy = confusion_matrix_data.copy()


def row_entropy(row):
    total = 0
    entropy = 0
    for i in range(len(confusion_matrix_data.columns)):
        total = total + row[i];
    for j in range(len(confusion_matrix_data.columns)):
        if (row[j] == 0):
            continue;
        entropy = entropy + row[j] / total * math.log2(row[j] / total)
    return -entropy


confusion_matrix_copy['Total'] = confusion_matrix_data.sum(axis=1)
confusion_matrix_copy['Row_entropy'] = confusion_matrix_data.apply(lambda row: row_entropy(row), axis=1)
total_data = confusion_matrix_copy['Total'].sum()
confusion_matrix_copy['entropy_prob'] = confusion_matrix_copy['Total'] / total_data * confusion_matrix_copy[
    'Row_entropy']
entropy_kmeans = confusion_matrix_copy['entropy_prob'].sum()

confusion_matrix_copy['Max_val'] = confusion_matrix_data.max(axis=1)
KMeans_purity_data = confusion_matrix_copy['Max_val'].sum() / total_data;

dbscan_feature = carb_feature_extraction.copy()[['BWZ Carb Input (grams)', 'mean CGM data']]

dbscan_data_feature_arr = dbscan_feature.values.astype('float32', copy=False)

dbscan_data_scaler = StandardScaler().fit(dbscan_data_feature_arr)
dbscan_data_feature_arr = dbscan_data_scaler.transform(dbscan_data_feature_arr)
dbscan_data_feature_arr

model = DBSCAN(eps=0.19, min_samples=5).fit(dbscan_data_feature_arr)

outliers_df = dbscan_feature[model.labels_ == -1]
clusters_df = dbscan_feature[model.labels_ != -1]


carb_feature_extraction['cluster'] = model.labels_

colors = model.labels_
colors_clusters = colors[colors != -1]
color_outliers = 'black'


clusters = Counter(model.labels_)

dbscana = dbscan_feature.values.astype('float32', copy=False)

bins_clusters_df_dbscan = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'dbscan_labels': list(model.labels_)},
                                       columns=['ground_true_arr', 'dbscan_labels'])


confusion_matrix_dbscan = pd.pivot_table(bins_clusters_df_dbscan, index='ground_true_arr', columns='dbscan_labels',
                                         aggfunc=len)
confusion_matrix_dbscan.fillna(value=0, inplace=True)
confusion_matrix_dbscan = confusion_matrix_dbscan.reset_index()
confusion_matrix_dbscan = confusion_matrix_dbscan.drop(columns=['ground_true_arr'])
confusion_matrix_dbscan = confusion_matrix_dbscan.drop(columns=[-1])
confusion_matrix_dbscan_copy = confusion_matrix_dbscan.copy()


def row_entropy_dbscan(row):
    total = 0
    entropy = 0
    for i in range(len(confusion_matrix_dbscan.columns)):
        total = total + row[i];

    for j in range(len(confusion_matrix_dbscan.columns)):
        if (row[j] == 0):
            continue;
        entropy = entropy + row[j] / total * math.log2(row[j] / total)
    return -entropy


confusion_matrix_dbscan_copy['Total'] = confusion_matrix_dbscan.sum(axis=1)
confusion_matrix_dbscan_copy['Row_entropy'] = confusion_matrix_dbscan.apply(lambda row: row_entropy_dbscan(row), axis=1)
total_data = confusion_matrix_dbscan_copy['Total'].sum()
confusion_matrix_dbscan_copy['entropy_prob'] = confusion_matrix_dbscan_copy['Total'] / total_data * \
                                               confusion_matrix_dbscan_copy['Row_entropy']
DBScan_entropy = confusion_matrix_dbscan_copy['entropy_prob'].sum()

confusion_matrix_dbscan_copy['Max_val'] = confusion_matrix_dbscan.max(axis=1)
DBSCAN_purity = confusion_matrix_dbscan_copy['Max_val'].sum() / total_data;

carb_feature_extraction = carb_feature_extraction.loc[carb_feature_extraction['cluster'] != -1]

dbscan_feature_extraction_centroid = carb_feature_extraction.copy()
centroid_carb_input_obj = {}
centroid_cgm_mean_obj = {}
squared_error = {}
DBSCAN_SSE = 0
for i in range(len(confusion_matrix_dbscan.columns)):
    cluster_group = carb_feature_extraction.loc[carb_feature_extraction['cluster'] == i]
    centroid_carb_input = cluster_group['BWZ Carb Input (grams)'].mean()
    centroid_cgm_mean = cluster_group['mean CGM data'].mean()
    centroid_carb_input_obj[i] = centroid_carb_input
    centroid_cgm_mean_obj[i] = centroid_cgm_mean

def centroid_carb_input_calc(row):
    return centroid_carb_input_obj[row['cluster']]

def centroid_cgm_mean_calc(row):
    return centroid_cgm_mean_obj[row['cluster']]

dbscan_feature_extraction_centroid['centroid_carb_input'] = carb_feature_extraction.apply(
    lambda row: centroid_carb_input_calc(row), axis=1)
dbscan_feature_extraction_centroid['centroid_cgm_mean'] = carb_feature_extraction.apply(
    lambda row: centroid_cgm_mean_calc(row), axis=1)

dbscan_feature_extraction_centroid['centroid_difference'] = 0

for i in range(len(dbscan_feature_extraction_centroid)):
    dbscan_feature_extraction_centroid['centroid_difference'].iloc[i] = math.pow(
        dbscan_feature_extraction_centroid['BWZ Carb Input (grams)'].iloc[i] -
        dbscan_feature_extraction_centroid['centroid_carb_input'].iloc[i], 2) + math.pow(
        dbscan_feature_extraction_centroid['mean CGM data'].iloc[i] -
        dbscan_feature_extraction_centroid['centroid_cgm_mean'].iloc[i], 2)
for i in range(len(confusion_matrix_dbscan.columns)):
    squared_error[i] = dbscan_feature_extraction_centroid.loc[dbscan_feature_extraction_centroid['cluster'] == i][
        'centroid_difference'].sum()

for i in squared_error:
    DBSCAN_SSE = DBSCAN_SSE + squared_error[i];

KMeans_DBSCAN = [KMeans_sse, DBSCAN_SSE, entropy_kmeans, DBScan_entropy, KMeans_purity_data, DBSCAN_purity]
print_df = pd.DataFrame(KMeans_DBSCAN).T
print_df
print_df.to_csv('Results.csv', header=False, index=False)
