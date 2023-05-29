# -*- coding: utf-8 -*-
"""raw2feature_EEG(removeOutlier).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nMHAQrzYdcYsEHNAH2SddGaqeuyRwO0k
"""

import numpy as np
from numpy import linalg as LA
import os
import pandas as pd

# def transferFormat(df : pd.DataFrame):
#     columns = df.columns
#     newCol = []
#     for index in df.index:
#         newCol += ['%d_%s'%(index, col) for col in columns]

#     tmp = df.to_numpy().reshape(1, -1)
#     df_new = pd.DataFrame(tmp, columns=newCol)
#     return df_new


def removeOutlier(listData, max_deviations=3):
    dataMean = np.mean(listData)
    dataStd = np.std(listData)
    distance_from_mean = abs(listData - dataMean)

    not_outlier_mask = distance_from_mean < max_deviations * dataStd
    not_outlier_data = listData[not_outlier_mask]

    return not_outlier_data


def getFeatures(data, col):
    featuretypes = ['min', 'max', 'mean', 'std']
    column_name = f'{col}_Data'
    sensor_cols = ['103', '104', '107', '108', '201', '202']
    result = pd.DataFrame()

    for sensor in sensor_cols:
        sensor_df = data[data['Value.EEGSensorType'].astype(str) == sensor]
        sensor_data = removeOutlier(sensor_df[column_name])
        sensor_cal = [np.min(sensor_data), np.max(sensor_data), np.mean(sensor_data), np.std(sensor_data)]
        sensor_result = pd.DataFrame([sensor_cal], columns=[
                                     f'{sensor}_{col}_{featuretype}' for featuretype in featuretypes])

        result = pd.concat([result, sensor_result], axis=1)

    return result


def calStageEEGFeatures(data):
    # if interferenceType == "":
    #     data = data[data["GlobalData._stage"] != "None"]
    # elif interferenceType == "NoInf":
    #     data = data[data["GlobalData._stage"] == "None"]
    # elif interferenceType == "All":
    #     pass

    # Remove data that have no signals
    data = data[(data != 0).all(axis=1)]

    cols = ['Value.DELTA', 'Value.THETA', 'Value.ALPHA', 'Value.BETA', 'Value.GAMMA']

    result = pd.DataFrame()

    for col in cols:
        dfFeatures = getFeatures(data, col)
        result = pd.concat([result, dfFeatures], axis=1)

    return result

if __name__ == '__main__':
    result = pd.DataFrame()
    data = pd.read_excel("D:\\ASD\\【Vive】特徵計算\\Pre\\ASD001_Pre\\puzzle\\EEGFeatureIndexesData_Puzzle.xlsx", sheet_name='_Post_001_EEGFeatureIndexesData')
    EEGFeatures = calStageEEGFeatures(data)
    result = pd.concat([result, EEGFeatures], axis = 0)
    print(result)
    result.to_excel("D:\\ASD\\【Vive】特徵計算\\Pre\\ASD001_Pre\\puzzle\\EV_EEG_Puzzle_All.xlsx",index=None)