
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

# CSV read
X = pd.read_csv('Siheung_Unc.csv')

target = X.iloc[:,1:]

imputer = KNNImputer(n_neighbors=3) #KNN
Y= imputer.fit_transform(target)

X.iloc[:,1:] = Y

X.to_csv('Siheung_Conc_Unc_210519.csv', index=False)

