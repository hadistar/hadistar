import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

#dir = 'D:/OneDrive - SNU/data/Chamber test_first_210423'
dir = 'D:/OneDrive - SNU/data/Chamber test_second_210424'
flow_dir = os.path.join(dir,'Flow/')
rh_t_dir = os.path.join(dir,'Temp_Humidity/')

rh_t_files = os.listdir(rh_t_dir)
flow_files = os.listdir(flow_dir)

data_RH_T = pd.DataFrame()
for i in rh_t_files:
    data_RH_T=data_RH_T.append(pd.read_csv(rh_t_dir+i))


data_RH_T['Time'] = pd.to_datetime(data_RH_T['Time'])


data_RH_T = data_RH_T.sort_values(by="Time")


plt.figure()
plt.plot(data_RH_T['Time'],data_RH_T['HumidityPV'],'ro', markersize=3, label='Humidity PV')
plt.plot(data_RH_T['Time'],data_RH_T[' HumiditySV'],'bo', markersize=3, label='Humidity SV')
plt.plot(data_RH_T['Time'],data_RH_T[' Temperature'],'ko', markersize=3, label='Temperature PV')
plt.gcf().autofmt_xdate()
plt.xlabel('Time (month-day hour)')
plt.ylabel('Temp or RH')
plt.legend()
plt.show()


filtered_df =data_RH_T.loc[data_RH_T["Time"].
    between('2021-04-24 15:00', '2021-04-25 08:00')]

t_0 = filtered_df['Time'].iloc[0]
elapsed_time = pd.to_timedelta(filtered_df['Time'] - t_0).astype('timedelta64[s]')/60

plt.figure()
plt.plot(elapsed_time,filtered_df['HumidityPV'],'ro-', markersize=3, label='Humidity')
plt.plot(elapsed_time,filtered_df[' HumiditySV'],'b-', markersize=3, label='Humidity set value')
#plt.plot(filtered_df['Time'],filtered_df[' Temperature'],'ko-', markersize=3, label='Temperature')
#plt.gcf().autofmt_xdate()
plt.xlabel('Elapsed time (minutes)')
plt.ylabel('Relative humidity (%)')
plt.grid(True, linestyle='--')
#plt.ylim([20,50])
#plt.xlim([0,84])
plt.legend(loc='upper right')
plt.show()




filtered_df =data_RH_T.loc[data_RH_T["Time"].
    between('2021-04-23 16:20', '2021-04-23 17:47')]

t_0 = filtered_df['Time'].iloc[0]
elapsed_time = pd.to_timedelta(filtered_df['Time'] - t_0).astype('timedelta64[s]')/60

plt.figure()
plt.plot(elapsed_time,filtered_df['HumidityPV'],'ro-', markersize=3, label='Humidity')
plt.plot(elapsed_time,filtered_df[' HumiditySV'],'b-', markersize=3, label='Humidity set value')
#plt.plot(filtered_df['Time'],filtered_df[' Temperature'],'ko-', markersize=3, label='Temperature')
#plt.gcf().autofmt_xdate()
plt.xlabel('Elapsed time (minutes)')
plt.ylabel('Relative humidity (%)')
plt.grid(True, linestyle='--')
plt.ylim([20,50])
plt.xlim([0,84])
plt.legend(loc='lower right')
plt.show()


filtered_df =data_RH_T.loc[data_RH_T["Time"].
    between('2021-04-23 14:20', '2021-04-23 16:00')]

t_0 = filtered_df['Time'].iloc[0]
elapsed_time = pd.to_timedelta(filtered_df['Time'] - t_0).astype('timedelta64[s]')/60

plt.figure()
plt.plot(elapsed_time,filtered_df['HumidityPV'],'ro-', markersize=3, label='Humidity')
plt.plot(elapsed_time,filtered_df[' HumiditySV'],'b-', markersize=3, label='Humidity set value')
#plt.plot(filtered_df['Time'],filtered_df[' Temperature'],'ko-', markersize=3, label='Temperature')
#plt.gcf().autofmt_xdate()
plt.xlabel('Elapsed time (minutes)',horizontalalignment='center')
plt.ylabel('Relative humidity (%)')
plt.xticks(rotation=0, ha='center')
plt.xlim([0,100])
plt.ylim([10,30])
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()


filtered_df =data_RH_T.loc[data_RH_T["Time"].
    between('2021-04-23 12:00', '2021-04-23 15:00')]

t_0 = filtered_df['Time'].iloc[0]
elapsed_time = pd.to_timedelta(filtered_df['Time'] - t_0).astype('timedelta64[s]')/60

plt.figure()
plt.plot(elapsed_time,filtered_df['HumidityPV'],'ro-', markersize=3, label='Humidity')
plt.plot(elapsed_time,filtered_df[' HumiditySV'],'b-', markersize=3, label='Humidity set value')
#plt.plot(filtered_df['Time'],filtered_df[' Temperature'],'ko-', markersize=3, label='Temperature')
#plt.gcf().autofmt_xdate()
plt.xlabel('Elapsed time (minutes)')
plt.ylabel('Relative humidity (%)')
plt.xticks(rotation=0)
plt.xlim([0,90])
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))
plt.ylim([5,30])
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()


'''
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

'''