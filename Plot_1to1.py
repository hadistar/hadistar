import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

import math
import numpy as np
from sklearn import linear_model
import sklearn

target_num = -1

x = np.array(data_elementals_answers.iloc[:,target_num])
y = np.array(data_elementals_predicted.iloc[:,target_num])

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

plt.figure()
#plt.scatter(x, y, s=40, facecolors='none', edgecolors='k')
plt.plot(x, y, 'ko', markersize=0.9, label=elementals[target_num])

plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,math.ceil(max(x.max(),y.max()))],[0,math.ceil(max(x.max(),y.max()))], 'k--')
plt.xlabel('Observed concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.ylabel('Predicted concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.text(x.max() * 0.1, y.max() * 0.6,
         'Y=%0.2fx+%0.3f\n$R^2$ = %0.2f (n=%s)'
         % (coef, intercept, r_squared, format(len(x), ',')))
#plt.axis([0, math.ceil(max(x.max(),y.max())), 0, math.ceil(max(x.max(),y.max()))])
plt.axis([0, 0.12,0,0.12])
plt.grid(True, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()