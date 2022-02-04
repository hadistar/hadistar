import pandas as pd
import os
from math import cos, sin, asin, sqrt, radians

def calc_distance(row):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1 = 37.3468122
    lon1 = 126.7400834
    lat2 = row['Latitude']
    lon2 = row['Longitude']

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

df1 = pd.DataFrame()
df2 = pd.DataFrame()

AirKorea_2019_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2019')
AirKorea_2020_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2020')


for i in range(len(AirKorea_2019_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2019\\'+AirKorea_2019_list[i])
    df1 = df1.append(temp)

for i in range(len(AirKorea_2020_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2020\\'+AirKorea_2020_list[i])
    df2 = df2.append(temp)


df = pd.read_csv('D:\\OneDrive - SNU\\data\\AirKorea_20191103.csv', encoding='euc-kr')

lat1 = 37.3468122
lon1 = 126.7400834

df['distance'] = df.apply(calc_distance, axis=1)
df = df.loc[df.distance<=100]


temp = df1.append(df2)

temp['temp'] = temp['측정일시'] - 1
temp['date'] = pd.to_datetime(temp['temp'], format='%Y%m%d%H')
temp['date'] = temp['date'] + pd.DateOffset(hours=1)

data = pd.DataFrame()

for i, row in df.iterrows():
    target = temp.loc[temp['측정소코드']==row['측정소코드']]
    target = target.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
    target['lat'] = row['Latitude']
    target['lon'] = row['Longitude']
    target['address'] = row['주소']
    target['distance'] = row['distance']

    data = data.append(target)

data.to_csv('AirKora_2019_2020_SH_100km.csv', encoding='euc-kr')

import pandas as pd

df = pd.read_csv('AirKora_2019_2020_SH_100km.csv')

df['date'] = pd.to_datetime(df['date'])

df2 = df.groupby(pd.Grouper(freq='Y', key='date'), 'Station code').mean()

df.loc[df['date']=='2020-06-26'].to_csv('AirKora_2019_2020_SH_100km_20200626.csv')

df = pd.read_excel('PM2.5_mean_day.xlsx')
df2 = pd.read_csv('AirKora_2019_2020_SH_100km.csv')
df2 = df2.loc[df2['date']=='2020-12-01']
df2 = df2[['Station code', 'lon', 'lat']]

data = pd.merge(df, df2, how='left', on='Station code')

data.to_csv('PM25_mean_day.csv', index=False)

df = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')





for i in range(10):
    print(i/10, df2['pm25'].quantile(i/10))

df2['pm25'].quantile(0.32)

0.32
0.75
0.992

(df2.quantile(0.32)*1.5).to_clipboard()

(df2.quantile(0.75)*1.5).to_clipboard()

(df2.quantile(0.992)*1.5).to_clipboard()


df2.loc[df2['해염 입자']>=2.67]









from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

import os
import requests


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace("%20", " ")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


urlTicker = Request('https://hpaudiobooks.club/hp-and-goblet-of-fire-audio-book/10/', headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(urlTicker).read()

soup = BeautifulSoup(webpage, 'html.parser')

t = soup.find_all('audio')
tt= BeautifulSoup(str(t)).find_all('a')

for i in tt:
    temp = i.get_text('href')
    download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면")








# book 5
urlTicker = Request('https://hpaudiobooks.club/half-blood-prince-audio-book-stephen-fry/', headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(urlTicker).read()

soup = BeautifulSoup(webpage, 'html.parser')

t = soup.find_all('audio')
tt= BeautifulSoup(str(t)).find_all('a')

for i in tt:
    temp = i.get_text('href')
    download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면\\5")


for pp in range(2,11):
    print(pp)

    urlTicker = Request('https://hpaudiobooks.club/audio-order-of-phoenix-stephen-fry/'+str(pp)+'/', headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(urlTicker).read()

    soup = BeautifulSoup(webpage, 'html.parser')

    t = soup.find_all('audio')
    tt= BeautifulSoup(str(t)).find_all('a')

    for i in tt:
        temp = i.get_text('href')
        download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면\\5")



# book 6

urlTicker = Request('https://hpaudiobooks.club/half-blood-prince-audio-book-stephen-fry/8/',
                    headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(urlTicker).read()

soup = BeautifulSoup(webpage, 'html.parser')

t = soup.find_all('audio')
tt = BeautifulSoup(str(t)).find_all('a')

for i in tt:
    temp = i.get_text('href')
    download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면\\6")

for pp in range(2, 9):
    print(pp)

    urlTicker = Request('https://hpaudiobooks.club/half-blood-prince-audio-book-stephen-fry/' + str(pp) + '/',
                        headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(urlTicker).read()

    soup = BeautifulSoup(webpage, 'html.parser')

    t = soup.find_all('audio')
    tt = BeautifulSoup(str(t)).find_all('a')

    for i in tt:
        temp = i.get_text('href')
        download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면\\6")


# book 7


urlTicker = Request('https://hpaudiobooks.club/book-7-harry-potter-and-the-deathly-hallows-stephen-fry-audiobook/',
                    headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(urlTicker).read()

soup = BeautifulSoup(webpage, 'html.parser')

t = soup.find_all('audio')
tt = BeautifulSoup(str(t)).find_all('a')

for i in tt:
    temp = i.get_text('href')
    download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면\\7")

for pp in range(2, 11):
    print(pp)

    urlTicker = Request('https://hpaudiobooks.club/book-7-harry-potter-and-the-deathly-hallows-stephen-fry-audiobook/' + str(pp) + '/',
                        headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(urlTicker).read()

    soup = BeautifulSoup(webpage, 'html.parser')

    t = soup.find_all('audio')
    tt = BeautifulSoup(str(t)).find_all('a')

    for i in tt:
        temp = i.get_text('href')
        download(temp, dest_folder="D:\OneDrive - SNU\바탕 화면\\7")




df = pd.read_csv('스마트시티_매핑결과_서울대_raw.csv')

locs = []

locations = pd.read_csv("D:\OneDrive - SNU\바탕 화면\locations.csv", encoding='euc-kr').dropna()

df2 = df.loc[df['date']=='2019-11-16'].copy()

def dist_(row,lat1,lon1):
    # lat1=37.34
    # lon1=126.76
    val = (row['lat']-lat1)**2 + (row['lon']-lon1)**2
    return val


for i in range(len(locations)):
    df2['dist']=df2.apply(dist_, args=(locations['위도'][i],locations['경도'][i]), axis=1)
    df2 = df2.sort_values(by='dist').reset_index(drop=True)
    locs.append(df2['location No.'][0])


locations['location No.'] = locs

df3 = pd.DataFrame()

for i in range(len(locations)):
    temp = df.loc[df['location No.']==locations['location No.'][i]].copy()
    temp2 = pd.merge(temp,locations, how='inner', on='location No.')
    temp3 = temp2[['date', '지점코드', '행정동', '위도', '경도','해염 입자', '석탄 연소', '기타 연소',
          '산업 배출', '토양', '2차 질산염', '2차 황산염', '자동차']].copy()
    df3 = df3.append(temp3)

df3 = df3.reset_index(drop=True)
df3.to_csv('서울대 시각화_행정동별.csv', encoding='euc-kr', index=False)

df3 = pd.read_csv('서울대 시각화_행정동별.csv', encoding='euc-kr')

df3.to_csv('서울대 시각화_행정동별_.csv', encoding='utf-8-sig', index=False)

df = pd.read_csv('스마트시티_매핑결과_서울대_raw.csv')
df.to_csv('스마트시티_매핑결과_서울대_raw_2.csv',encoding='utf-8-sig', index=False)



# <2021-11-24> box plot

df = pd.read_csv("Smartcity_BSMRM_8Locations.csv").dropna()

import matplotlib.pyplot as plt

plt.figure()
df.boxplot(column=["NO3-"], by=['StationNo'])
plt.show()
plt.close()



import pandas as pd

df = pd.read_csv("D:\OneDrive - SNU\바탕 화면\등농도맵핑 샘플 데이터.csv")

locs = df.drop_duplicates(['spot_lat', 'spot_lon'])

SNU = pd.read_csv('스마트시티_매핑결과_서울대_raw_2.csv').drop_duplicates(['lat','lon'])

df1 = pd.DataFrame(locs)
df2 = pd.DataFrame(SNU)

df1['point'] = [(x, y) for x,y in zip(df1['spot_lat'], df1['spot_lon'])]
df2['point'] = [(x, y) for x,y in zip(df2['lat'], df2['lon'])]

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]

from scipy.spatial.distance import cdist

df1['closest'] = [closest_point(x, list(df2['point'])) for x in df1['point']]

df1['location No.'] = [match_value(df2, 'point', x, 'location No.') for x in df1['closest']]

df = pd.read_csv('스마트시티_매핑결과_서울대_raw_2.csv')

dates = df.date.drop_duplicates()

ans = pd.DataFrame()
for d in dates:
    temp = df.loc[df['date']==d]
    temp2 = pd.merge(df1, temp, how='inner', on='location No.')
    ans = ans.append(temp2)

final = ans[['spot_cd', 'spot_lat', 'spot_lon', 'date', '해염 입자', '석탄 연소', '기타 연소',
       '산업 배출', '토양', '2차 질산염', '2차 황산염', '자동차']].copy()


final.to_csv('스마트시티_매핑결과_서울대_raw_좌표재설정_211215.csv', index=False, encoding='euc-kr')



# 2022-01-07, heatmap, BNFA vs. PMF

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dir = 'D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\220107_comparison_BNFA_PMF_daejeon\\'

df1 = pd.read_csv(dir+'BNFA_Daejeon_q6.csv')
df2 = pd.read_csv(dir+'PMF_Daejeon_q6.csv')

df1 = df1.add_prefix('BNFA_')
df2 = df2.add_prefix('PMF_')

df = pd.concat([df1, df2], axis=1)

plt.figure(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr, cmap='bwr', annot=True)
plt.show()
