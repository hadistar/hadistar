import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython
import keras_tuner as kt
import shutil
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)

Data_Name = ['4_AP+Meteo_3_Ulsan']

Random_State = [777, 1004, 322]

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

col_dic = {'ocec': ocec, 'elementals': elementals, 'ions': ions, 'ocec-elementals': ocec+elementals,
           'ions-ocec': ions+ocec, 'ions-elementals': ions+elementals, 'ions-ocec-elementals': ions+ocec+elementals}
Missing_Col = ['ions','ocec', 'elementals', 'ocec-elementals', 'ions-ocec', 'ions-elementals', 'ions-ocec-elementals']


