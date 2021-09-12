
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



# 2021-09-10 kNN imputing for BNFA in Siheung (Smartcity project)

import pandas as pd
import os
from sklearn.impute import KNNImputer
import numpy as np

os.chdir('D:\Dropbox\Bayesian modeling\Young Su Lee')

X = pd.read_csv("SH_AP_SO2_yslee_210903.csv")
target = X.iloc[:,1:]
imputer = KNNImputer(n_neighbors=3) #KNN
Y= imputer.fit_transform(np.array(target))

X.iloc[:,1:] = Y

X.to_csv("SH_AP_SO2_kNN_yslee_210910.csv", index=False)