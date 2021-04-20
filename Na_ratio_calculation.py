import pandas as pd
import os

df_Seoul = pd.read_csv('D:/OneDrive - SNU/data/PM25speciationdata/preprocessed/Seoul_preprocessed_withoutKNN_hourly.csv')
df_BN = pd.read_csv('D:/OneDrive - SNU/data/PM25speciationdata/preprocessed/Baengnyeong_preprocessed_withoutKNN_hourly.csv')
df_DJ = pd.read_csv('D:/OneDrive - SNU/data/PM25speciationdata/preprocessed/Daejeon_preprocessed_withoutKNN_hourly.csv')
df_GJ = pd.read_csv('D:/OneDrive - SNU/data/PM25speciationdata/preprocessed/Gwangju_preprocessed_withoutKNN_hourly.csv')
df_Jeju = pd.read_csv('D:/OneDrive - SNU/data/PM25speciationdata/preprocessed/Jeju_preprocessed_withoutKNN_hourly.csv')
df_US = pd.read_csv('D:/OneDrive - SNU/data/PM25speciationdata/preprocessed/Ulsan_preprocessed_withoutKNN_hourly.csv')

df  = df_US
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})