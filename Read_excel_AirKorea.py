import pandas as pd
import os


#df = pd.DataFrame()
#for file in os.listdir('./data/2019'):
#    df = df.append(pd.read_excel('./data/2019/'+file))

df = pd.read_excel('./data/2019/2019년 11월.xlsx')

df['date'] = df['측정일시'].astype(str).str[0:-2]
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
test = df.groupby(['측정소코드','date'], as_index=False).mean()

test = pd.merge(test, df[['측정소코드','주소']].drop_duplicates(),
                on='측정소코드', how='left')

temp = test.loc[test['date'] == '2019-11-02']

temp.to_csv('AirKorea_20191103.csv',index=False)
