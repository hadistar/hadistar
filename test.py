import pandas as pd

data_filter = pd.read_csv('D:/OneDrive - SNU/data/filterpm25conc_6sites.csv', encoding='euc-kr')

data_filter = data_filter.loc[data_filter['ID']=='Seoul']

data_filter = data_filter.to_datetime