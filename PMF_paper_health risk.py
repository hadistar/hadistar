import pandas as pd

def ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF):
    return (C*Ing_R*EF*ED*CF/(BW*AT1))

def ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF):
    return (C*SA*AF*ABS*EF*ED*CF/(BW*AT1))

def ADD_inh(C,IR,ET,EF,ED,AT2):
    return(C*IR*ET*EF*ED/AT2)

def ILCR(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, IR, ET, AT2, IUR, SF0, GIABS):
    temp = ADD_inh(C, IR, ET, EF, ED, AT2) * IUR +\
           ADD_derm(C,SA,AF,ABS,EF,ED,BW,AT1,CF) * SF0/GIABS +\
           ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF) * SF0
    return(temp)

def HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, IR, ET, AT2, RfCi, RfD0, GIABS):
    temp = ADD_inh(C, IR, ET, EF, ED, AT2) / (RfCi*1000) + \
           ADD_derm(C, SA, AF, ABS, EF, ED, BW, AT1, CF) / (RfD0*GIABS) + \
           ADD_ing(C, Ing_R, EF, ED, BW, AT1, CF) * SF0/RfD0
    return (temp)

# Data loading

df_total = pd.read_excel("D:\\Dropbox\\PMF_paper\\PMF results_raw\\PMF results_YSLEE_X_건강평가용_v8.xlsx", sheet_name='X_total')
df_total = df_total.iloc[:,:24]
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


# Condition setting

gender = 'male'
element = 'Cr'
# factors = ['Ing_R', 'EF','ED','BW','SA','AF','ABS','ET','AT1','AT2','CF', 'IR']

# Variable selection

Ing_R = factor_basics.loc[0][gender]
EF = factor_basics.loc[1][gender]
ED = factor_basics.loc[2][gender]
BW = factor_basics.loc[3][gender]
SA = factor_basics.loc[4][gender]
AF = factor_basics.loc[5][gender]
ABS = factor_basics.loc[6][gender]
ET = factor_basics.loc[7][gender]
AT1 = factor_basics.loc[8][gender]
AT2 = factor_basics.loc[9][gender]
CF = factor_basics.loc[10][gender]
IR = factor_basics.loc[11][gender] # IR 체크 필요

RfD0 = factor_elements.loc[0][element]
RfCi = factor_elements.loc[1][element]
GIABS = factor_elements.loc[2][element]
IUR = factor_elements.loc[3][element]
SF0 = factor_elements.loc[4][element]

C = df_total.Cr

ILCR(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, IR, ET, AT2, IUR, SF0, GIABS).mean()


HQ(C, Ing_R, EF, ED, BW, AT1, CF, SA, AF, ABS, IR, ET, AT2, RfCi, RfD0, GIABS)



