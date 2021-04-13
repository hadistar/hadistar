import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

from sklearn import linear_model
import sklearn

# Data loading
data_filter = pd.read_csv('D:/OneDrive - SNU/data/filterpm25conc_6sites.csv', encoding='euc-kr')
data_auto = pd.read_csv("D:/OneDrive - SNU/data/Seoul_preprocessed_withoutKNN_daily.csv")

# Data indexing
data_filter = data_filter.loc[data_filter['ID']=='Seoul']

# to datetime type
data_filter['date'] = pd.to_datetime(data_filter['date'], format="%m/%d/%Y", exact = False)
data_auto['date'] = pd.to_datetime(data_auto['date'], format="%Y-%m-%d", exact = False)

# 정리
data_filter = data_filter.drop(['ID'], axis=1)
data_auto = data_auto.drop(['location','lat','lon','Si','Ba','PM10'], axis=1)
data_auto.columns = data_filter.columns

# Merging
data = pd.merge(data_filter,data_auto, how='inner', on='date')

# Comparison

r2s = []
names = []

for i in range(25):

    temp = data.iloc[:,[i+1,i+27]].dropna()
    x = temp.iloc[:,0]
    y = temp.iloc[:,1]

    # Create linear regression object
    linreg = linear_model.LinearRegression()
    # Fit the linear regression model
    model = linreg.fit(x.to_numpy().reshape(-1,1), y.to_numpy().reshape(-1,1))
    # Get the intercept and coefficients
    intercept = model.intercept_
    coef = model.coef_
    result = [intercept, coef]
    predicted_y = x.to_numpy().reshape(-1,1)*coef + intercept
    r_squared = sklearn.metrics.r2_score(y,predicted_y)
    plt.figure(i+1)
    plt.scatter(x, y, c='k', s=5.0)
    plt.plot(x, predicted_y, 'b-', 0.1)
    plt.xlabel('Filter data, ' + data.columns[i+1][:-2] + ' (ug/m3)')
    plt.ylabel('Automated measuring data, '+data.columns[i+27][:-2] + ' (ug/m3)')
    plt.text(x.max()*0.5, y.max()*0.85, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
    plt.axis([0,temp.max().max(),0,temp.max().max()])
    plt.grid(True)
    plt.savefig('Seoul_'+data.columns[i+27][:-2]+'.png')
    plt.show()
    r2s.append(r_squared)
    names.append(data.columns[i+27][:-2])

pd.DataFrame(names).to_clipboard()
pd.DataFrame(r2s).to_clipboard()
