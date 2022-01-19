
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time.csv')

plt.figure(figsize=(40,40))
corr = df.corr()
sns.heatmap(corr, cmap='bwr', annot=True)
plt.savefig('AWMA_EDA_corr_1st.png')
plt.show()


# drop PM25, groundtemp, 30cmtemp, dewpoint

df=df.drop(columns=['PM25', 'groundtemp', '30cmtemp','dewpoint'])