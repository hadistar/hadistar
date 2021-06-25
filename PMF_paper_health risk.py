import pandas as pd
import numpy as np
import scipy.stats


def ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF):
    temp = (C*Ing_R*EF*ED*CF/(BW*AT1))*1e-6
    return temp

def ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF):
    temp = (C*SA*AF*ABS*EF*ED*CF/(BW*AT1))*1e-6
    return temp

def ADD_inh(C,ET,EF,ED,AT2):
    return(C*ET*EF*ED/AT2)

def ILCR(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS):
    temp = ADD_inh(C, ET, EF, ED, AT2) * IUR +\
           ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF) * SF0/GIABS +\
           ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF) * SF0
    return(temp)

def HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS):
    temp = ADD_inh(C, ET, EF, ED, AT2) / (RfCi*1000) + \
           ADD_derm(C, SA, AF, ABS, EF, ED, BW, AT1, CF) / (RfD0*GIABS) + \
           ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF)/RfD0
    return (temp)


def ILCR_ing(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS):
    temp = ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF) * SF0
    return(temp)
def ILCR_derm(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS):
    temp = ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF) * SF0/GIABS
    return(temp)
def ILCR_inh(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS):
    temp = ADD_inh(C, ET, EF, ED, AT2) * IUR
    return(temp)

def HQ_ing(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS):
    temp = ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF)/RfD0
    return (temp)
def HQ_derm(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS):
    temp = ADD_derm(C, SA, AF, ABS, EF, ED, BW, AT1, CF) / (RfD0*GIABS)
    return (temp)
def HQ_inh(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS):
    temp = ADD_inh(C, ET, EF, ED, AT2) / (RfCi*1000)
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

df_total['Cr_6'] = df_total['Cr'] *0.3
df_total['Cr_3'] = df_total['Cr'] *0.7

df_salts['Cr_6'] = df_salts['Cr'] *0.3
df_salts['Cr_3'] = df_salts['Cr'] *0.7

df_soil['Cr_6'] = df_soil['Cr'] *0.3
df_soil['Cr_3'] = df_soil['Cr'] *0.7

df_SS['Cr_6'] = df_SS['Cr'] *0.3
df_SS['Cr_3'] = df_SS['Cr'] *0.7

df_coal['Cr_6'] = df_coal['Cr'] *0.3
df_coal['Cr_3'] = df_coal['Cr'] *0.7

df_biomass['Cr_6'] = df_biomass['Cr'] *0.3
df_biomass['Cr_3'] = df_biomass['Cr'] *0.7

df_indus_smelting['Cr_6'] = df_indus_smelting['Cr'] *0.3
df_indus_smelting['Cr_3'] = df_indus_smelting['Cr'] *0.7

df_indus_oil['Cr_6'] = df_indus_oil['Cr'] *0.3
df_indus_oil['Cr_3'] = df_indus_oil['Cr'] *0.7

df_heating['Cr_6'] = df_heating['Cr'] *0.3
df_heating['Cr_3'] = df_heating['Cr'] *0.7

df_SN['Cr_6'] = df_SN['Cr'] *0.3
df_SN['Cr_3'] = df_SN['Cr'] *0.7

df_mobile['Cr_6'] = df_mobile['Cr'] *0.3
df_mobile['Cr_3'] = df_mobile['Cr'] *0.7

factor_basics = pd.read_excel("D:\\Dropbox\\PMF_paper\\건강영향파트\\Health_variables_yslee.xlsx", sheet_name='Basics')
factor_elements = pd.read_excel("D:\\Dropbox\\PMF_paper\\건강영향파트\\Health_variables_yslee.xlsx", sheet_name='Elements')
factor_elements = factor_elements.iloc[:6,:]

# Condition setting
df = {'total':df_total, 'salts':df_salts, 'soil':df_soil, 'SS':df_SS, 'coal':df_coal, 'biomass':df_biomass,
      'indus_smelting':df_indus_smelting, 'indus_oil': df_indus_oil, 'heating': df_heating, 'SN':df_SN, 'mobile':df_mobile}

gender = 'average'
elements = ['As','Cr_6', 'Cr_3','Cu','Ni','Pb','Zn','V','Mn']
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

        temp = [str(name) + ", " + str(element)]
        for i in range(len(C)):
            temp.append(ILCR(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
        for i in range(len(C)):
            temp.append(HQ(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

        results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
    temp = results.sum()
    temp['Type'] = "**Sum, " + str(name)
    results = results.append(temp, ignore_index=True)
    #results.to_excel('health_risk_result_v8.xlsx', sheet_name=name, index=False)



    a = a.append(results)




# Original data for uncertainty analysis

df_orig = pd.read_excel('(201105)시흥시입력자료_재계산농도로대체2_NABlank.xlsx')
df_orig['Cr_6'] = df_orig['Cr'] *0.3
df_orig['Cr_3'] = df_orig['Cr'] *0.7
results = pd.DataFrame()
data = df_orig
name = 'original'
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

    temp = [str(name) + ", " + str(element)]
    for i in range(len(C)):
        temp.append(ILCR(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
    for i in range(len(C)):
        temp.append(HQ(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

    results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
temp = results.sum()
temp['Type'] = "**Sum, " + str(name)
results = results.append(temp, ignore_index=True)
a = a.append(results)

a.to_csv('health_v8_210624_yslee.csv', index=False)



# inh

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

        temp = [str(name) + ", " + str(element)]
        for i in range(len(C)):
            temp.append(ILCR_inh(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
        for i in range(len(C)):
            temp.append(HQ_inh(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

        results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
    temp = results.sum()
    temp['Type'] = "**Sum, " + str(name)
    results = results.append(temp, ignore_index=True)
    a = a.append(results)

df_orig = pd.read_excel('(201105)시흥시입력자료_재계산농도로대체2_NABlank.xlsx')
df_orig['Cr_6'] = df_orig['Cr'] *0.3
df_orig['Cr_3'] = df_orig['Cr'] *0.7
results = pd.DataFrame()
data = df_orig
name = 'original'
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

    temp = [str(name) + ", " + str(element)]
    for i in range(len(C)):
        temp.append(ILCR_inh(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
    for i in range(len(C)):
        temp.append(HQ_inh(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

    results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
temp = results.sum()
temp['Type'] = "**Sum, " + str(name)
results = results.append(temp, ignore_index=True)
a = a.append(results)

a.to_csv('health_v8_210624_yslee_inh.csv', index=False)



# derm

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

        temp = [str(name) + ", " + str(element)]
        for i in range(len(C)):
            temp.append(ILCR_derm(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
        for i in range(len(C)):
            temp.append(HQ_derm(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

        results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
    temp = results.sum()
    temp['Type'] = "**Sum, " + str(name)
    results = results.append(temp, ignore_index=True)
    a = a.append(results)

df_orig = pd.read_excel('(201105)시흥시입력자료_재계산농도로대체2_NABlank.xlsx')
df_orig['Cr_6'] = df_orig['Cr'] *0.3
df_orig['Cr_3'] = df_orig['Cr'] *0.7
results = pd.DataFrame()
data = df_orig
name = 'original'
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

    temp = [str(name) + ", " + str(element)]
    for i in range(len(C)):
        temp.append(ILCR_derm(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
    for i in range(len(C)):
        temp.append(HQ_derm(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

    results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
temp = results.sum()
temp['Type'] = "**Sum, " + str(name)
results = results.append(temp, ignore_index=True)
a = a.append(results)

a.to_csv('health_v8_210624_yslee_derm.csv', index=False)



# derm

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

        temp = [str(name) + ", " + str(element)]
        for i in range(len(C)):
            temp.append(ILCR_ing(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
        for i in range(len(C)):
            temp.append(HQ_ing(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

        results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
    temp = results.sum()
    temp['Type'] = "**Sum, " + str(name)
    results = results.append(temp, ignore_index=True)
    a = a.append(results)

df_orig = pd.read_excel('(201105)시흥시입력자료_재계산농도로대체2_NABlank.xlsx')
df_orig['Cr_6'] = df_orig['Cr'] *0.3
df_orig['Cr_3'] = df_orig['Cr'] *0.7
results = pd.DataFrame()
data = df_orig
name = 'original'
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

    temp = [str(name) + ", " + str(element)]
    for i in range(len(C)):
        temp.append(ILCR_ing(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
    for i in range(len(C)):
        temp.append(HQ_ing(C[i], Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

    results = results.append(pd.DataFrame([temp], columns=columns), ignore_index=True)
temp = results.sum()
temp['Type'] = "**Sum, " + str(name)
results = results.append(temp, ignore_index=True)
a = a.append(results)

a.to_csv('health_v8_210624_yslee_ing.csv', index=False)





# All time-series values by source

health=pd.DataFrame()

for name in df:
    print(name)
    health_source = pd.DataFrame()
    for row in df[name].iterrows():
#        print(row[0])
        rowlist = {}
        temp_sum = 0
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
            C = row[1][element]

            head = str(name) + ", " + str(element)
            temp_ILCR = ILCR(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS)
            rowlist[head+' ILCR'] = temp_ILCR
            if temp_ILCR > 0:
                temp_sum += temp_ILCR
        rowlist[name+', ILCR_sum'] = temp_sum
        temp_sum = 0
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
            C = row[1][element]

            head = str(name) + ", " + str(element)
            temp_HQ = HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS)
            rowlist[head+' HQ'] = temp_HQ
            temp_sum += temp_HQ

        rowlist[name+', HQ_sum'] = temp_sum

        aa = pd.DataFrame.from_dict(rowlist, orient='index').T
        health_source = health_source.append(aa)

    health = pd.concat([health, health_source], axis=1)

health = health.reset_index(drop=True)
health['date'] = df_total['Date_Time']

health.to_csv('health_v8_all time_210624.csv')

# end of all time calculation


# For Seoul and Daebu
df_Daebu = pd.read_csv('D:\\Dropbox\\PMF_paper\\건강영향파트\\Seoul_Daebu_raw.csv').loc[0]
df_Seoul = pd.read_csv('D:\\Dropbox\\PMF_paper\\건강영향파트\\Seoul_Daebu_raw.csv').loc[1]

df_Daebu['Cr_6'] = df_Daebu['Cr'] *0.3
df_Daebu['Cr_3'] = df_Daebu['Cr'] *0.7

df_Seoul['Cr_6'] = df_Seoul['Cr'] *0.3
df_Seoul['Cr_3'] = df_Seoul['Cr'] *0.7


a = pd.DataFrame()
data = df_Daebu
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
    C = data[element]

    temp = ["Daebu, " + str(element)]
    temp.append(ILCR(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
    temp.append(HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

    results = results.append(pd.DataFrame([temp]), ignore_index=True)
temp = results.sum()
temp[0] = "**Sum, Daebu"
results = results.append(temp, ignore_index=True)



a = pd.DataFrame()
data = df_Seoul

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
    C = data[element]

    temp = ["Seoul, " + str(element)]
    temp.append(ILCR(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, IUR, SF0, GIABS))
    temp.append(HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, ET, AT2, RfCi, RfD0, GIABS))

    results = results.append(pd.DataFrame([temp]), ignore_index=True)
temp = results.sum()
temp[0] = "**Sum, Seoul"
results = results.append(temp, ignore_index=True)

results.to_csv('HR_Seoul_Daebu.csv', index=False)