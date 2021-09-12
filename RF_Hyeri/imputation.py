import numpy as np
import pandas as pd
import os

# directory and data import
os.getcwd()
#os.chdir('D:\\GitHub\\Code\\py_study_21S\\imputation')
df = pd.read_csv("Seoul_preprocessed_withoutKNN_hourly.csv")

# Keep an untouched copy for later
df_orig = df.copy()
df= df.dropna(axis=0)
df_test = df.copy()
df.head(2)
df.columns

# Generate unique lists of random integers
import random
np.random.seed(10)
inds1 = list(set(np.random.choice(len(df),526, replace=False)))
inds2 = list(set(np.random.choice(len(df),526, replace=False)))

# Replace the values at given index position with NaNs
df['OC'] = [val if i not in inds1 else np.nan for i, val in enumerate(df['OC'])]
df['EC'] = [val if i not in inds2 else np.nan for i, val in enumerate(df['EC'])]

# Get count of missing values by column
df.isnull().sum()

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

# Make an instance and perform the imputation
imputer = MissForest()
x = df.drop(df.columns[[0, 1, 2, 3]], axis=1)
x_imputed = imputer.fit_transform(x)

data_array = x_imputed
column_names = ['PM2.5', 'SO42.', 'NO3.', 'Cl.',
       'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.', 'OC', 'EC', 'Si', 'S', 'K', 'Ca',
       'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba',
       'Pb', 'PM10']
result = pd.DataFrame(data_array, columns=column_names)
result.isnull().sum()

# Add imputed values as columns to the untouched dataset
df_test['MF_OC'] = x_imputed[:, 9]
df_test['MF_EC'] = x_imputed[:, 10]
comparison_df = df_test[['OC', 'MF_OC', 'EC', 'MF_EC']]

# Calculate absolute errors
comparison_df['ABS_ERROR_OC'] = np.abs(comparison_df['OC'] - comparison_df['MF_OC'])
comparison_df['ABS_ERROR_EC'] = np.abs(comparison_df['EC'] - comparison_df['MF_EC'])
comparison_df.head()

# Show only rows where imputation was performed
imputation_result_OC = comparison_df.iloc[sorted([*inds1])]
imputation_result_EC = comparison_df.iloc[sorted([*inds2])]

imputation_result_OC.to_csv("imputation_result_OC.csv")
imputation_result_EC.to_csv("imputation_result_EC.csv")

x =  x_imputed[inds1,9]
y = df_test.iloc[inds1]
y = y['OC']

from sklearn import linear_model
import sklearn


x = np.array(x)
y = np.array(y)

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
