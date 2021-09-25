import pandas as pd
import numpy as np

data_insulin = pd.read_csv("InsulinData.csv", low_memory=False)
data_cgm = pd.read_csv(
    "CGMData.csv", low_memory=False, usecols=["Date", "Time", "Sensor Glucose (mg/dL)"]
)

data_cgm["date_time_stamp"] = pd.to_datetime(data_cgm["Date"] + " " + data_cgm["Time"])

removedData = data_cgm[data_cgm["Sensor Glucose (mg/dL)"].isna()]["Date"].unique()

data_cgm = data_cgm.set_index("Date").drop(index=removedData).reset_index()

copy_data_cgm = data_cgm.copy()

copy_data_cgm = copy_data_cgm.set_index(pd.DatetimeIndex(data_cgm["date_time_stamp"]))

data_insulin["date_time_stamp"] = pd.to_datetime(
    data_insulin["Date"] + " " + data_insulin["Time"]
)

begin_autoMode = (
    data_insulin.sort_values(by="date_time_stamp", ascending=True)
    .loc[data_insulin["Alarm"] == "AUTO MODE ACTIVE PLGM OFF"]
    .iloc[0]["date_time_stamp"]
)

data_autoMode = data_cgm.sort_values(by="date_time_stamp", ascending=True).loc[
    data_cgm["date_time_stamp"] >= begin_autoMode
]

data_manualMode = data_cgm.sort_values(by="date_time_stamp", ascending=True).loc[
    data_cgm["date_time_stamp"] < begin_autoMode
]

new_data_autoMode = data_autoMode.copy()

new_data_autoMode = new_data_autoMode.set_index("date_time_stamp")

autoMode_list = (
    new_data_autoMode.groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    .where(lambda x: x > 0.8 * 288)
    .dropna()
    .index.tolist()
)

new_data_autoMode = new_data_autoMode.loc[new_data_autoMode["Date"].isin(autoMode_list)]

automode_hyperglycemia_overnight_time = (
    new_data_autoMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_hyperglycemia_daytime_time = (
    new_data_autoMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_hyperglycemia_wholeday_time = (
    new_data_autoMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

automode_critical_hyperglycemia_overnight_time = (
    new_data_autoMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_critical_hyperglycemia_daytime_time = (
    new_data_autoMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_critical_hyperglycemia_wholeday_time = (
    new_data_autoMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

automode_normal_overnight_time = (
    new_data_autoMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_autoMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_autoMode["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_normal_daytime_time = (
    new_data_autoMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_autoMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_autoMode["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_normal_wholeday_time = (
    new_data_autoMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_autoMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_autoMode["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

automode_normal_overnight_time_2 = (
    new_data_autoMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_autoMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_autoMode["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_normal_daytime_time_2 = (
    new_data_autoMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_autoMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_autoMode["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_normal_wholeday_time_2 = (
    new_data_autoMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_autoMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_autoMode["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

automode_hypoglycemia_lvl1_overnight_time = (
    new_data_autoMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_hypoglycemia_lvl1_daytime_time = (
    new_data_autoMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_hypoglycemia_lvl1_wholeday_time = (
    new_data_autoMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

automode_hypoglycemia_lvl2_overnight_time = (
    new_data_autoMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_hypoglycemia_lvl2_daytime_time = (
    new_data_autoMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
automode_hypoglycemia_lvl2_wholeday_time = (
    new_data_autoMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_autoMode["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

new_data_manualMode = data_manualMode.copy()
new_data_manualMode = new_data_manualMode.set_index("date_time_stamp")

manualMode_list = (
    new_data_manualMode.groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    .where(lambda x: x > 0.8 * 288)
    .dropna()
    .index.tolist()
)

new_data_manualMode = new_data_manualMode.loc[
    new_data_manualMode["Date"].isin(manualMode_list)
]

manualMode_hyperglycemia_overnight_time = (
    new_data_manualMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_hyperglycemia_daytime_time = (
    new_data_manualMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_hyperglycemia_wholeday_time = (
    new_data_manualMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] > 180]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

manualMode_critical_hyperglycemia_overnight_time = (
    new_data_manualMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_critical_hyperglycemia_daytime_time = (
    new_data_manualMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_critical_hyperglycemia_wholeday_time = (
    new_data_manualMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] > 250]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

manualMode_normal_overnight_time = (
    new_data_manualMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_manualMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_manualMode["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_normal_daytime_time = (
    new_data_manualMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_manualMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_manualMode["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_normal_wholeday_time = (
    new_data_manualMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_manualMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_manualMode["Sensor Glucose (mg/dL)"] <= 180)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

manualMode_normal_overnight_time_2 = (
    new_data_manualMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_manualMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_manualMode["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_normal_daytime_time_2 = (
    new_data_manualMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_manualMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_manualMode["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_normal_wholeday_time_2 = (
    new_data_manualMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[
        (new_data_manualMode["Sensor Glucose (mg/dL)"] >= 70)
        & (new_data_manualMode["Sensor Glucose (mg/dL)"] <= 150)
    ]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

manualMode_hypoglycemia_lvl1_overnight_time = (
    new_data_manualMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_hypoglycemia_lvl1_daytime_time = (
    new_data_manualMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_hypoglycemia_lvl1_wholeday_time = (
    new_data_manualMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] < 70]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

manualMode_hypoglycemia_lvl2_overnight_time = (
    new_data_manualMode.between_time("0:00:00", "05:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_hypoglycemia_lvl2_daytime_time = (
    new_data_manualMode.between_time("6:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)
manualMode_hypoglycemia_lvl2_wholeday_time = (
    new_data_manualMode.between_time("0:00:00", "23:59:59")[
        ["Date", "Time", "Sensor Glucose (mg/dL)"]
    ]
    .loc[new_data_manualMode["Sensor Glucose (mg/dL)"] < 54]
    .groupby("Date")["Sensor Glucose (mg/dL)"]
    .count()
    / 288
    * 100
)

manualMode_days_count = len(manualMode_list)

manualMode_hyperglycemia_overnight = round(
    (sum(manualMode_hyperglycemia_overnight_time) / manualMode_days_count), 2
)
manualMode_critical_hyperglycemia_overnight = round(
    (sum(manualMode_critical_hyperglycemia_overnight_time) / manualMode_days_count), 2
)
manualMode_normal_overnight = round(
    (sum(manualMode_normal_overnight_time) / manualMode_days_count), 2
)
manualMode_normal_overnight_2 = round(
    (sum(manualMode_normal_overnight_time_2) / manualMode_days_count), 2
)
manualMode_hypoglycemia_lvl1_overnight = round(
    (sum(manualMode_hypoglycemia_lvl1_overnight_time) / manualMode_days_count), 2
)
manualMode_hypoglycemia_lvl2_overnight = round(
    (sum(manualMode_hypoglycemia_lvl2_overnight_time) / manualMode_days_count), 2
)

manualMode_hyperglycemia_daytime = round(
    (sum(manualMode_hyperglycemia_daytime_time) / manualMode_days_count), 2
)
manualMode_critical_hyperglycemia_daytime = round(
    (sum(manualMode_critical_hyperglycemia_daytime_time) / manualMode_days_count), 2
)
manualMode_normal_daytime = round(
    (sum(manualMode_normal_daytime_time) / manualMode_days_count), 2
)
manualMode_normal_daytime_2 = round(
    (sum(manualMode_normal_daytime_time_2) / manualMode_days_count), 2
)
manualMode_hypoglycemia_lvl1_daytime = round(
    (sum(manualMode_hypoglycemia_lvl1_daytime_time) / manualMode_days_count), 2
)
manualMode_hypoglycemia_lvl2_daytime = round(
    (sum(manualMode_hypoglycemia_lvl2_daytime_time) / manualMode_days_count), 2
)

manualMode_hyperglycemia_wholeday = round(
    (sum(manualMode_hyperglycemia_wholeday_time) / manualMode_days_count), 2
)
manualMode_critical_hyperglycemia_wholeday = round(
    (sum(manualMode_critical_hyperglycemia_wholeday_time) / manualMode_days_count), 2
)
manualMode_normal_wholeday = round(
    (sum(manualMode_normal_wholeday_time) / manualMode_days_count), 2
)
manualMode_normal_wholeday_2 = round(
    (sum(manualMode_normal_wholeday_time_2) / manualMode_days_count), 2
)
manualMode_hypoglycemia_lvl1_wholeday = round(
    (sum(manualMode_hypoglycemia_lvl1_wholeday_time) / manualMode_days_count), 2
)
manualMode_hypoglycemia_lvl2_wholeday = round(
    (sum(manualMode_hypoglycemia_lvl2_wholeday_time) / manualMode_days_count), 2
)

autoMode_days_count = len(autoMode_list)

automode_hyperglycemia_overnight = round(
    (sum(automode_hyperglycemia_overnight_time) / autoMode_days_count), 2
)
automode_critical_hyperglycemia_overnight = round(
    (sum(automode_critical_hyperglycemia_overnight_time) / autoMode_days_count), 2
)
automode_normal_overnight = round(
    (sum(automode_normal_overnight_time) / autoMode_days_count), 2
)
automode_normal_overnight_2 = round(
    (sum(automode_normal_overnight_time_2) / autoMode_days_count), 2
)
automode_hypoglycemia_lvl1_overnight = round(
    (sum(automode_hypoglycemia_lvl1_overnight_time) / autoMode_days_count), 2
)
automode_hypoglycemia_lvl2_overnight = round(
    (sum(automode_hypoglycemia_lvl2_overnight_time) / autoMode_days_count), 2
)

automode_hyperglycemia_daytime = round(
    (sum(automode_hyperglycemia_daytime_time) / autoMode_days_count), 2
)
automode_critical_hyperglycemia_daytime = round(
    (sum(automode_critical_hyperglycemia_daytime_time) / autoMode_days_count), 2
)
automode_normal_daytime = round(
    (sum(automode_normal_daytime_time) / autoMode_days_count), 2
)
automode_normal_daytime_2 = round(
    (sum(automode_normal_daytime_time_2) / autoMode_days_count), 2
)
automode_hypoglycemia_lvl1_daytime = round(
    (sum(automode_hypoglycemia_lvl1_daytime_time) / autoMode_days_count), 2
)
automode_hypoglycemia_lvl2_daytime = round(
    (sum(automode_hypoglycemia_lvl2_daytime_time) / autoMode_days_count), 2
)

automode_hyperglycemia_wholeday = round(
    (sum(automode_hyperglycemia_wholeday_time) / autoMode_days_count), 2
)
automode_critical_hyperglycemia_wholeday = round(
    (sum(automode_critical_hyperglycemia_wholeday_time) / autoMode_days_count), 2
)
automode_normal_wholeday = round(
    (sum(automode_normal_wholeday_time) / autoMode_days_count), 2
)
automode_normal_wholeday_2 = round(
    (sum(automode_normal_wholeday_time_2) / autoMode_days_count), 2
)
automode_hypoglycemia_lvl1_wholeday = round(
    (sum(automode_hypoglycemia_lvl1_wholeday_time) / autoMode_days_count), 2
)
automode_hypoglycemia_lvl2_wholeday = round(
    (sum(automode_hypoglycemia_lvl2_wholeday_time) / autoMode_days_count), 2
)

columnsName = [
    "Modes",
    "Over Night Percentage time in hyperglycemia (CGM > 180 mg/dL)",
    "Over Night percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
    "Over Night percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
    "Over Night percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
    "Over Night percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
    "Over Night percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
    "Day Time Percentage time in hyperglycemia (CGM > 180 mg/dL)",
    "Day Time percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
    "Day Time percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
    "Day Time percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
    "Day Time percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
    "Day Time percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
    "Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)",
    "Whole Day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
    "Whole Day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
    "Whole Day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
    "Whole Day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
    "Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
]

results = pd.DataFrame(columns=columnsName)
results["Modes"] = ["Manual Mode", "Auto Mode"]

results["Over Night Percentage time in hyperglycemia (CGM > 180 mg/dL)"] = [
    manualMode_hyperglycemia_overnight,
    automode_hyperglycemia_overnight,
]
results["Over Night percentage of time in hyperglycemia critical (CGM > 250 mg/dL)"] = [
    manualMode_critical_hyperglycemia_overnight,
    automode_critical_hyperglycemia_overnight,
]
results[
    "Over Night percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)"
] = [manualMode_normal_overnight, automode_normal_overnight]
results[
    "Over Night percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)"
] = [manualMode_normal_overnight_2, automode_normal_overnight_2]
results["Over Night percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)"] = [
    manualMode_hypoglycemia_lvl1_overnight,
    automode_hypoglycemia_lvl1_overnight,
]
results["Over Night percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"] = [
    manualMode_hypoglycemia_lvl2_overnight,
    automode_hypoglycemia_lvl2_overnight,
]

results["Day Time Percentage time in hyperglycemia (CGM > 180 mg/dL)"] = [
    manualMode_hyperglycemia_daytime,
    automode_hyperglycemia_daytime,
]
results["Day Time percentage of time in hyperglycemia critical (CGM > 250 mg/dL)"] = [
    manualMode_critical_hyperglycemia_daytime,
    automode_critical_hyperglycemia_daytime,
]
results["Day Time percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)"] = [
    manualMode_normal_daytime,
    automode_normal_daytime,
]
results[
    "Day Time percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)"
] = [manualMode_normal_daytime_2, automode_normal_daytime_2]
results["Day Time percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)"] = [
    manualMode_hypoglycemia_lvl1_daytime,
    automode_hypoglycemia_lvl1_daytime,
]
results["Day Time percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"] = [
    manualMode_hypoglycemia_lvl2_daytime,
    automode_hypoglycemia_lvl2_daytime,
]

results["Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)"] = [
    manualMode_hyperglycemia_wholeday,
    automode_hyperglycemia_wholeday,
]
results["Whole Day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)"] = [
    manualMode_critical_hyperglycemia_wholeday,
    automode_critical_hyperglycemia_wholeday,
]
results["Whole Day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)"] = [
    manualMode_normal_wholeday,
    automode_normal_wholeday,
]
results[
    "Whole Day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)"
] = [manualMode_normal_wholeday_2, automode_normal_wholeday_2]
results["Whole Day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)"] = [
    manualMode_hypoglycemia_lvl1_wholeday,
    automode_hypoglycemia_lvl1_wholeday,
]
results["Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"] = [
    manualMode_hypoglycemia_lvl2_wholeday,
    automode_hypoglycemia_lvl2_wholeday,
]

# add extra column for gradescope issue
extracolumn = [1.1, 1.1]

results = pd.concat([results, pd.DataFrame(extracolumn)], axis=1)
results.set_index("Modes", inplace=True)
results.to_csv("Results.csv", index=False, header=None)
