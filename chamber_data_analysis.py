import pandas as pd
import os
import matplotlib.pyplot as plt

dir = 'D:/OneDrive - SNU/data/Chamber test_first_210423'
flow_dir = os.path.join(dir,'Flow/')
rh_t_dir = os.path.join(dir,'Temp_Humidity/')

rh_t_files = os.listdir(rh_t_dir)
flow_files = os.listdir(flow_dir)

data_RH_T = pd.DataFrame()
for i in rh_t_files:
    data_RH_T=data_RH_T.append(pd.read_csv(rh_t_dir+i))

data_RH_T['Time'] = pd.to_datetime(data_RH_T['Time'])

plt.figure()
plt.plot(data_RH_T['Time'],data_RH_T['HumidityPV'],'ro', markersize=3, label='Humidity PV')
plt.plot(data_RH_T['Time'],data_RH_T[' HumiditySV'],'bo', markersize=3, label='Humidity SV')
plt.plot(data_RH_T['Time'],data_RH_T[' Temperature'],'ko', markersize=3, label='Temperature PV')
plt.gcf().autofmt_xdate()
plt.xlabel('Time (month-day hour)')
plt.ylabel('Temp or RH')
plt.legend()
plt.show()


data_Flow = pd.DataFrame()
for i in flow_files:
    data_Flow=data_Flow.append(pd.read_csv(flow_dir+i))

data_Flow['Time'] = pd.to_datetime(data_Flow['Time'])

plt.figure()
plt.plot(data_Flow['Time'],data_Flow['Dry Air'],'ro', markersize=3, label='Dry Air Flow (mL/min)')
plt.plot(data_Flow['Time'],data_Flow[' Wet Air'],'bo', markersize=3, label='Wet Air Flow (mL/min)')
plt.plot(data_Flow['Time'],data_Flow[' Output Flow'],'ko', markersize=3, label='Output Flow (mL/min)')
plt.gcf().autofmt_xdate()
plt.xlabel('Time (month-day hour)')
plt.ylabel('Flow rate (mL/min)')
plt.legend()
plt.show()

