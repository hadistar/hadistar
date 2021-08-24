import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams.update({'figure.autolayout': True})

import math
import numpy as np
from sklearn import linear_model
import sklearn


df = pd.read_csv('data\\SH_sampler_vs.AirKorea_210719.csv')
x = np.array(df.sampler)
y = np.array(df.AirKorea)

target_num = -1

#x = np.array(data_elementals_answers.iloc[:,target_num])
#y = np.array(data_elementals_predicted.iloc[:,target_num])

# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(x.reshape(-1,1), y.reshape(-1,1))


#model = linreg.fit(x.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1))
# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure(figsize=(5,5))
#plt.scatter(x, y, s=40, facecolors='none', edgecolors='k')
plt.plot(x, y, 'ro', markersize=8, mfc='none')

plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,110],[0,110], 'k--')
#plt.plot([0,math.ceil(max(x.max(),y.max()))],[0,math.ceil(max(x.max(),y.max()))], 'k--')
plt.xlabel('Concentration of sampled filter ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.ylabel('Concentration of AirKorea ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.text(x.max() * 0.1, y.max() * 0.6,
         'y = %0.2fx + %0.2f\n$r^2$ = %0.2f (n=%s)'
         % (coef, intercept, r_squared, format(len(x), ',')))
#plt.axis([0, math.ceil(max(x.max(),y.max())), 0, math.ceil(max(x.max(),y.max()))])
plt.axis([0, 110,0,110])
plt.grid(True, linestyle='--')
#plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


date = pd.to_datetime(df.date)


plt.figure(figsize=(8,5))
plt.plot(date,y,'k-', label='AirKorea', linewidth=1.2)
plt.plot(date, x, 'b--', label='Sampled filter', linewidth=1.2)

plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.xlabel('Date (year-month)')
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim([0,110])
#plt.tight_layout()
plt.xticks(rotation=45)
plt.show()