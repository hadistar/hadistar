import pandas as pd
import os
import numpy as np

#os.chdir('Python_study_2021_summer')

df = pd.read_excel('D:\hadistar\Python_study_2021_summer\Chlorite, Chlorate IC data (자동 저장됨).xlsx', sheet_name="BW PAC.csv")

df = df.iloc[:,1:4]

import numpy as np

x = np.array([70, 100, 150, 250, 500])
x = np.repeat(x, [3], axis=0)

x = x.reshape(-1,1)

df['conc'] = 0
df['conc'] = x


avg = df.groupby(['conc']).mean().reset_index()
std = df.groupby(['conc']).apply(np.std)
std.pop('conc')

avg = avg.reset_index()
std = std.reset_index()

std *= 10

#avg.conc = pd.to_numeric(avg.conc, errors='coerce')
#std.conc = pd.to_numeric(std.conc, errors='coerce')

#avg = avg.sort_values(by='conc')
#std = std.sort_values(by='conc')


import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14



x, y = np.array(avg.conc), np.array(avg.chlorite)

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
#plt.plot(avg.conc, avg.chlorite, 'bo')
#plt.plot(avg.conc, avg.chlorate, 'ro')
plt.plot(x, predicted_y, 'b-', 0.1)

plt.errorbar(avg.conc, avg.chlorite,
             marker='o',
             color='C0',
             yerr=std.chlorite,
             markersize=10,
             linestyle="")

plt.errorbar(avg.conc, avg.chlorate,
             marker='o',
             color='C9',
             yerr=std.chlorite,
             markersize=10,
             linestyle="")
plt.show()


