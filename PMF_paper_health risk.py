import pandas as pd
import numpy as np
import scipy.stats


def ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF):
    return (C*Ing_R*EF*ED*CF/(BW*AT1))

def ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF):
    return (C*SA*AF*ABS*EF*ED*CF/(BW*AT1))

def ADD_inh(C,ET=ET,EF=EF,ED=ED,AT2=AT2):
    return(C*ET*EF*ED/AT2)

def ILCR(C, Ing_R=Ing_R, EF=EF, ED=ED, BW=BW, AT1=AT1, CF=CF, SA=SA, AF=AF, ABS=ABS, ET=ET, AT2=AT2, IUR=IUR, SF0=SF0, GIABS=GIABS):
    temp = ADD_inh(C, ET, EF, ED, AT2) * IUR +\
           ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF) * SF0/GIABS +\
           ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF) * SF0
    return(temp)

def HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS):
    temp = ADD_inh(C, ET, EF, ED, AT2) / (RfCi*1000) + \
           ADD_derm(C, SA, AF, ABS, EF, ED, BW, AT1, CF) / (RfD0*GIABS) + \
           ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF)/RfD0
    return (temp)

def makeC(df):
    return ([df.mean(), df.quantile(q=0.05), df.quantile(q=0.25),df.quantile(q=0.5),df.quantile(q=0.75),df.quantile(q=0.95)])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# Data loading

df_total = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_total')
df_total = df_total.iloc[:,:25]
df_salts = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_salts')
df_soil = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_soil')
df_SS = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_SS')
df_coal = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_coal')
df_biomass = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_biomass')
df_indus_smelting = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_Indus_smelting')
df_indus_oil = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_Indus_oil')
df_heating = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_heating')
df_SN = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_SN')
df_mobile = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_mobile')

factor_basics = pd.read_excel("D:\\Dropbox\\PMF_paper\\건강영향파트\\Health_variables_yslee.xlsx", sheet_name='Basics')
factor_elements = pd.read_excel("D:\\Dropbox\\PMF_paper\\건강영향파트\\Health_variables_yslee.xlsx", sheet_name='Elements')
factor_elements = factor_elements.iloc[:6,:]

# Condition setting
df = {'total':df_total, 'salts':df_salts, 'soil':df_soil, 'SS':df_SS, 'coal':df_coal, 'biomass':df_biomass,
      'indus_smelting':df_indus_smelting, 'indus_oil': df_indus_oil, 'heating': df_heating, 'SN':df_SN, 'mobile':df_mobile}

gender = 'average'
elements = ['As','Cr','Cu','Ni','Pb','Zn','V','Mn']
# factors = ['Ing_R', 'EF','ED','BW','SA','AF','ABS','ET','AT1','AT2','CF', 'IR']
columns = ['mean','0.05','0.25', '0.5', '0.75','0.95']
# Variable selection

columns = ['Type','ILCR_mean','ILCR_0.05','ILCR_0.25', 'ILCR_0.5', 'ILCR_0.75','ILCR_0.95',
                   'HQ_mean','HQ_0.05','HQ_0.25', 'HQ_0.5', 'HQ_0.75','HQ_0.95']
a = pd.DataFrame()
for name in df:
    data = df[name]
    results = pd.DataFrame()
    for element in elements:
        Ing_R = factor_basics.loc[0][gender]
        EF = factor_basics.loc[1][gender]
        ED = factor_basics.loc[2][gender]
        BW = factor_basics.loc[3][gender]
        SA = factor_basics.loc[4][gender]
        AF = factor_basics.loc[5][gender]
        ET = factor_basics.loc[6][gender]
        AT1 = factor_basics.loc[7][gender]
        AT2 = factor_basics.loc[8][gender]
        CF = factor_basics.loc[9][gender]

        RfD0 = factor_elements.loc[0][element]
        RfCi = factor_elements.loc[1][element]
        GIABS = factor_elements.loc[2][element]
        IUR = factor_elements.loc[3][element]
        SF0 = factor_elements.loc[4][element]
        ABS = factor_elements.loc[5][element]
    #    results = results.append(pd.DataFrame(['total, ' + element], columns=['Type']))
        C = makeC(data[element])
        temp = [str(name) + "," + str(element)]
        for i in range(len(C)):
            temp.append(ILCR(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
        for i in range(len(C)):
            temp.append(HQ(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

        results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
    temp = results.sum()
    temp['Type'] = "Sum," + str(name)
    results = results.append(temp, ignore_index=True)
    #results.to_excel('health_risk_result_v8.xlsx', sheet_name=name, index=False)



    a = a.append(results)

a.to_csv('health_v8_210616_yslee.csv', index=False)


