import spidev, time
import serial
import struct
import time
import RPi.GPIO as GPIO
import sys,os
from datetime import datetime
from time import sleep
import csv
import board
from adafruit_htu21d import HTU21D

#ADC
spi_0 = spidev.SpiDev()
spi_0.open(0,0)
spi_0.max_speed_hz=500000

NOwe0 = 355
NOae0 = 337
NOs = 0.57
NOwez = 288
NOaez = 289

NO2we0 = 230
NO2ae0 = 236
NO2s = 0.280
NO2wez = 234
NO2aez = 236

OXwe0 = 217
OXae0 = 222
OXs = 0.293
OXwez = 224
OXaez = 226

COwe0 = 346
COae0 = 345
COs = 0.415
COwez = 350
COaez = 339
                                    

def readadc(adcnum):
    if adcnum >7 or adcnum<0:
        return -1
    r_0 =spi_0.xfer2([1 ,8 +adcnum << 4, 0])
    adcout_0 = ((r_0[1] & 3)<< 8) + r_0[2]
    return adcout_0


#HTU21D
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = HTU21D(i2c)

def tem():
    return sensor.temperature

def hum():
    return sensor.relative_humidity

f = open("co_cham.csv", "a")
f.write('datetime,temperature(C),humidity(%),NO_WE(mv),NO_AE(mV),NO2_WE(mV),NO2_AE(mV),OX_WE(mV),OX_AE(mV),CO_WE(mV),CO_AE(mV),NO(ppb),NO2(ppb),OX(ppb),CO(ppb)\n')
f.close()    
############edit#####################

while True:
    
    for i in range(0, 8):
        for j in range(0, 8):
            if i == 0: #NO, need correction
                NOwe = readadc(i) *(5000/1023)
                if j == 1:
                    NOae = readadc(j) *(5000/1023)
                    
                    ppb_conc_NO = ((NOwe - NOwez) - (NOwe0/NOae0)*(NOae - NOaez)) / NOs


                    
            else:
                if i == 2: #NO2
                    NO2we = readadc(i)*(5000/1023)
                    if j == 3:
                        NO2ae = readadc(j)*(5000/1023)
                        
                        ppb_conc_NO2 = ((NO2we - NO2wez) - (NO2ae - NO2aez)) / NO2s

                else:
                    if i == 4: #OX
                        OXwe = readadc(i)*(5000/1023)
                        if j == 5:
                            OXae = readadc(j)*(5000/1023)
                            
                            
                            ppb_conc_OX = ((OXwe - OXwez) - (OXae - OXaez)) / OXs
                            
                    else:
                        if i == 6: #CO
                            COwe = readadc(i)*(5000/1023)
                            if j == 7:
                                COae = readadc(j)*(5000/1023)
                                                                
                                ppb_conc_CO = ((COwe - COwez) - (COae - COaez)) / COs
                                #print(time.strftime('%y-%m-%d %H:%M:%S') +'|'+"Temperature = {0:0.1f}*C Humidity = {1:0.1f}%".format(t, h))                                
                                
                                
                                time.sleep(0.5)
                                print(time.strftime('%y-%m-%d %H:%M:%S'))
                                print("[T]: " + str(tem()) + ' | ' + "[H]: " + str(hum())) 
                                print("[NOwe]: "+ str(NOwe)+ "  [NOae]: "+ str(NOae) + "  [NO_ppb]: " + str(ppb_conc_NO))
                                print("[NO2we]: "+ str(NO2we)+ "  [NO2ae]: "+ str(NO2ae) +"  [NO2_ppb]: " + str(ppb_conc_NO2))
                                print("[OXwe]: "+ str(OXwe)+ "  [OXae]: "+ str(OXae) +"  [OX_ppb]: " + str(ppb_conc_OX))
                                print("[COwe]: "+ str(COwe)+ "  [COae]: "+ str(COae) +"  [CO_ppb]: " + str(ppb_conc_CO))        
                                print("=================================================================================")
                                
                                file = open("co_cham.csv", "a")
                                file.write(time.strftime('%Y/%m/%d %H:%M:%S') + ',' + str(tem()) + ',' + str(hum()) + ',' + str(NOwe)
                                           + ',' + str(NOae) + ',' + str(NO2we)+ ',' + str(NO2ae) + ',' + str(OXwe)+',' + str(OXae) +',' + str(COwe) + ',' + str(COae)
                                           + ',' + str(ppb_conc_NO) +',' + str(ppb_conc_NO2) +',' + str(ppb_conc_OX)+','+str(ppb_conc_CO)+'\n')
                                
                                 
                                    
                                #timestamp= time.strftime('%y-%m-%d %H:%M:%S')
                                #data = [str(ppm_conc_NO), str(ppm_conc_NO2), str(ppm_conc_OX), str(ppm_conc_CO)]
                                #for d in range(len(data)):
                                #    f.write(timestamp + '' +  data[d]+'\n')
                                file.close()
                                time.sleep(0.5)
                                
   



