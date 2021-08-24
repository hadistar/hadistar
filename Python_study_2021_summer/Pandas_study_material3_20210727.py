import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
import numpy as np
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

import os
os.getcwd()
os.chdir('D:/hadistar/Python_study_2021_summer')

df1 = pd.read_csv('210713_PM2.5_mean_day.csv')
df2 = pd.read_csv('210713_AirKorea_20191103.csv', encoding='euc-kr')
df2 = df2.rename(columns={'측정소코드':'Station code'})

df3 = pd.merge(df1, df2[['Station code', 'Latitude', 'Longitude']], how='inner', on='Station code')
df3.head(3)

df4 = df3.loc[df3['Station code']==111121]
df5 = df3.loc[df3['Station code']==131501]

data = pd.merge(df4,df5, how='inner', on='date')

# 1. Correlation between df4.PM25_mean and df5.PM25_mean

x = np.array(data.PM25_mean_x)
y = np.array(data.PM25_mean_y)

# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model

model = linreg.fit(x.reshape(-1,1), y.reshape(-1,1))
# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure()
plt.plot(x, y, 'ko', markersize=0.9, label='1:1 Plot')
plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,140],[0,140], 'k--')
plt.xlabel('Concentration (x) ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.ylabel('Concentration (y) ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.text(x.max() * 0.1, y.max() * 0.6,
         'Y=%0.2f*x+%0.2f\n$R^2$ = %0.2f (n=%s)'
         % (coef, intercept, r_squared, format(len(x), ',')))
plt.axis([0,140,0,140])
plt.grid(True, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 2. Curve fit

from scipy.optimize import curve_fit

x, y = data.date[8:15], data.PM25_mean_x[8:15]
x = pd.to_datetime(x)
x = np.array(x)
x = x - x[0]
x = x/np.timedelta64(1,'D')
y = np.array(y)

plt.figure()
plt.plot(x, y, 'ro', label='Observed data')
plt.ylim([0,150])
plt.legend()
plt.show()

# 2-1. Linear regression
def func1(x, a, b):
    return a * x + b

popt1, pcov1 = curve_fit(func1, x, y)
print(popt1)

plt.figure()
plt.plot(x, y, 'ro', label='Observed data')
plt.plot(x, func1(x, *popt1), 'k-',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt1))
plt.ylim([0,150])
plt.legend()
plt.show()

residuals = y-func1(x, *popt1)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

# 2-1. (Optional) Comparing with sklearn results

linreg = linear_model.LinearRegression()
model = linreg.fit(x.reshape(-1,1), y.reshape(-1,1))
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

# 2-2. Exp function
def func2(x, a, b, c):
    return a * np.exp(-b * x) + c

popt2, pcov2 = curve_fit(func2, x, y)
print(popt2)

plt.figure()
plt.plot(x, y, 'ro', label='Observed data')
plt.plot(x, func2(x, *popt2), 'k-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt2))
plt.ylim([0,150])
plt.legend()
plt.show()

residuals = y-func2(x, *popt2)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

# 2-3. Sigmoid function
def func3(x, a, b, c):
    return (a/(1+ np.exp(-b * x))) + c

popt3, pcov3 = curve_fit(func3, x, y)
print(popt3)

residuals = y- func3(x, *popt3)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

plt.figure()
plt.plot(x, y, 'ro', label='Observed data')
plt.plot(x, func3(x, *popt3), 'k-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3))
plt.ylim([0,150])
plt.legend()
plt.show()

# 3. Integration
from scipy.integrate import quad
# 3-1. Linear curve integration
I1 = quad(func1, 1, 2, args=tuple(popt1))
print(I1)

# 3-2. Sigmoid curve integration and visualization
I2 = quad(func3, 1, 2, args=tuple(popt3))

plt.figure()
plt.plot(x, y, 'ro', label='Observed data')
plt.plot(x, func3(x, *popt3), 'k-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3))
plt.fill_between([1,2],[func3(1,*popt3), func3(2,*popt3)], alpha=0.5)
plt.text(1.5,0.2*(func2(1,*popt3)+func3(2,*popt3)), 'Area=%2.2f' % I2[0],
         horizontalalignment='center', verticalalignment='center')
plt.ylim([0,150])
plt.legend()
plt.show()

### Integration by definition

x_list = np.linspace(1,2,10**5)
area = 0

for i, x_val in enumerate(x_list):
    if i==0:
        pass
    else:
        dx = x_list[i] - x_list[i-1]
        y_val = 0.5*(func1(x_list[i],*popt1)+func1(x_list[i-1],*popt1))
        area = area + dx*y_val
print(area)


pibo = [1,1]
for i in range(100):
    if i == 0:
        pass
    else:
        pibo.append(pibo[i]+pibo[i-1])



