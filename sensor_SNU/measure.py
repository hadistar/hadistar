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
                                    

def readadc_0(adcnum_0):
    if adcnum_0 >7 or adcnum_0<0:
        return -1
    r_0 =spi_0.xfer2([1 ,8 +adcnum_0 << 4, 0])
    adcout_0 = ((r_0[1] & 3)<< 8) + r_0[2]
    return adcout_0


#HTU21D
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = HTU21D(i2c)

def tem():
    return sensor.temperature

def hum():
    return sensor.relative_humidity

f = open("220220.csv", "a")
f.write('datetime,temperature(C),humidity(%),NO_WE(mv),NO_AE(mV),NO2_WE(mV),NO2_AE(mV),OX_WE(mV),OX_AE(mV),CO_WE(mV),CO_AE(mV),NO(ppb),NO2(ppb),OX(ppb),CO(ppb),PM2.5,PM10\n')
f.close()    


#PM7003
class PMS7003(object):
    # PMS7003 protocol data (HEADER 2byte + 30byte)
    PMS_7003_PROTOCOL_SIZE = 32
    
    # PMS7003 data list
    HEADER_HIGH = 0  # 0x42
    HEADER_LOW = 1  # 0x4d
    FRAME_LENGTH = 2  # 2x13+2(data+check bytes)
    DUST_PM1_0_CF1 = 3  # PM1.0 concentration unit μ g/m3（CF=1，standard particle）
    DUST_PM2_5_CF1 = 4  # PM2.5 concentration unit μ g/m3（CF=1，standard particle）
    DUST_PM10_0_CF1 = 5  # PM10 concentration unit μ g/m3（CF=1，standard particle）
    DUST_PM1_0_ATM = 6  # PM1.0 concentration unit μ g/m3（under atmospheric environment）
    DUST_PM2_5_ATM = 7  # PM2.5 concentration unit μ g/m3（under atmospheric environment）
    DUST_PM10_0_ATM = 8  # PM10 concentration unit μ g/m3  (under atmospheric environment)
    DUST_AIR_0_3 = 9  # indicates the number of particles with diameter beyond 0.3 um in 0.1 L of air.
    DUST_AIR_0_5 = 10  # indicates the number of particles with diameter beyond 0.5 um in 0.1 L of air.
    DUST_AIR_1_0 = 11  # indicates the number of particles with diameter beyond 1.0 um in 0.1 L of air.
    DUST_AIR_2_5 = 12  # indicates the number of particles with diameter beyond 2.5 um in 0.1 L of air.
    DUST_AIR_5_0 = 13  # indicates the number of particles with diameter beyond 5.0 um in 0.1 L of air.
    DUST_AIR_10_0 = 14  # indicates the number of particles with diameter beyond 10 um in 0.1 L of air.
    RESERVEDF = 15  # Data13 Reserved high 8 bits
    RESERVEDB = 16  # Data13 Reserved low 8 bits
    CHECKSUM = 17  # Checksum code


    f = open("220220.csv", "a")
    f.write('datetime,temperature(C),humidity(%),NO_WE(mv),NO_AE(mV),NO2_WE(mV),NO2_AE(mV),OX_WE(mV),OX_AE(mV),CO_WE(mV),CO_AE(mV),NO(ppb),NO2(ppb),OX(ppb),CO(ppb),PM2.5,PM10\n')
    
    # header check
    def header_chk(self, buffer):

        if (buffer[self.HEADER_HIGH] == 66 and buffer[self.HEADER_LOW] == 77):
            return True

        else:
            return False

    # chksum value calculation
    def chksum_cal(self, buffer):

        buffer = buffer[0:self.PMS_7003_PROTOCOL_SIZE]

        # data unpack (Byte -> Tuple (30 x unsigned char <B> + unsigned short <H>))
        chksum_data = struct.unpack('!30BH', buffer)

        chksum = 0

        for i in range(30):
            chksum = chksum + chksum_data[i]

        return chksum

    # checksum check
    def chksum_chk(self, buffer):

        chk_result = self.chksum_cal(buffer)

        chksum_buffer = buffer[30:self.PMS_7003_PROTOCOL_SIZE]
        chksum = struct.unpack('!H', chksum_buffer)

        if (chk_result == chksum[0]):
            return True

        else:
            return False

    # protocol size(small) check
    def protocol_size_chk(self, buffer):

        if (self.PMS_7003_PROTOCOL_SIZE <= len(buffer)):
            return True

        else:
            return False

    # protocol check
    def protocol_chk(self, buffer):

        if (self.protocol_size_chk(buffer)):

            if (self.header_chk(buffer)):

                if (self.chksum_chk(buffer)):

                    return True
                else:
                    print("Chksum err")
            else:
                print("Header err")
        else:
            print("Protol err")

        return False

        # unpack data

    # <Tuple (13 x unsigned short <H> + 2 x unsigned char <B> + unsigned short <H>)>
    def unpack_data(self, buffer):

        buffer = buffer[0:self.PMS_7003_PROTOCOL_SIZE]

        # data unpack (Byte -> Tuple (13 x unsigned short <H> + 2 x unsigned char <B> + unsigned short <H>))
        data = struct.unpack('!2B13H2BH', buffer)

        return data

    def print_serial(self, buffer):

        chksum = self.chksum_cal(buffer)
        data = self.unpack_data(buffer)
        
        NO_we = readadc_0(0)*(5000/1023)
        NO_ae = readadc_0(1)*(5000/1023)
        NO2_we = readadc_0(2)*(5000/1023)
        NO2_ae = readadc_0(3)*(5000/1023)
        OX_we = readadc_0(4)*(5000/1023)
        OX_ae = readadc_0(5)*(5000/1023)
        CO_we = readadc_0(6)*(5000/1023)
        CO_ae = readadc_0(7)*(5000/1023)
       
        NO_ppb = ((NO_we - NOwez) - (NOwe0/NOae0)*(NO_ae - NOaez)) / NOs
        NO2_ppb = ((NO2_we - NO2wez) - (NO2_ae  - NO2aez)) / NO2s
        OX_ppb = ((OX_we - OXwez) - (OX_ae - OXaez)) / OXs
        CO_ppb = ((CO_we  - COwez) - (CO_ae - COaez)) / COs
        
        

        print("============================================================================")
        print(time.strftime('%y-%m-%d %H:%M:%S'))
        #print("Header : %c %c \t\t | Frame length : %s" % (
        #data[self.HEADER_HIGH], data[self.HEADER_LOW], data[self.FRAME_LENGTH]))
        #print("PM 1.0 (CF=1) : %s\t | PM 1.0 : %s" % (data[self.DUST_PM1_0_CF1], data[self.DUST_PM1_0_ATM]))
        #print("PM 2.5 (CF=1) : %s\t | PM 2.5 : %s" % (data[self.DUST_PM2_5_CF1], data[self.DUST_PM2_5_ATM]))
        print("[PM 2.5] : %s" %(data[self.DUST_PM2_5_ATM]))
        #print("PM 10.0 (CF=1) : %s\t | PM 10.0 : %s" % (data[self.DUST_PM10_0_CF1], data[self.DUST_PM10_0_ATM]))
        print("[PM 10.0] : %s" %(data[self.DUST_PM10_0_ATM]))
        #print("0.3um in 0.1L of air : %s" % (data[self.DUST_AIR_0_3]))
        #print("0.5um in 0.1L of air : %s" % (data[self.DUST_AIR_0_5]))
        #print("1.0um in 0.1L of air : %s" % (data[self.DUST_AIR_1_0]))
        #print("2.5um in 0.1L of air : %s" % (data[self.DUST_AIR_2_5]))
        #print("5.0um in 0.1L of air : %s" % (data[self.DUST_AIR_5_0]))
        #print("10.0um in 0.1L of air : %s" % (data[self.DUST_AIR_10_0]))
        #print("Reserved F : %s | Reserved B : %s" % (data[self.RESERVEDF], data[self.RESERVEDB]))
        #print("CHKSUM : %s | read CHKSUM : %s | CHKSUM result : %s" % (
        #chksum, data[self.CHECKSUM], chksum == data[self.CHECKSUM]))
        print("[T]: " + str(tem()) + ' | ' + "[H]: " + str(hum())) 
        print("[NOwe]: "+ str(NO_we)+ "  [NOae]: "+ str(NO_ae) + "  [NO_ppb]: " + str(NO_ppb))
        print("[NO2we]: "+ str(NO2_we)+ "  [NOae]: "+ str(NO2_ae) +"  [NO2_ppb]: " + str(NO2_ppb))
        print("[OXwe]: "+ str(OX_we)+ "  [OXae]: "+ str(OX_ae) +"  [OX_ppb]: " + str(OX_ppb))
        print("[COwe]: "+ str(CO_we)+ "  [COae]: "+ str(CO_ae) +"  [CO_ppb]: " + str(CO_ppb))        
        print("============================================================================")
        f = open("220220.csv", "a")
        timestamp= time.strftime('%y-%m-%d %H:%M:%S')
        f.write(timestamp + ',' + str(tem()) + ',' + str(hum())
                #+','  + str(readadc_0(0)*(5000/1023))
                
                +','  + str(NO_we)                
                +','  + str(NO_ae)
                +','  + str(NO2_we)
                +','  + str(NO2_ae)
                +','  + str(OX_we)
                +','  + str(OX_ae)
                +','  + str(CO_we)
                +','  + str(CO_ae)
                + ',' + str(NO_ppb)
                + ',' + str(NO2_ppb)
                + ',' + str(OX_ppb)
                + ',' + str(CO_ppb)
                
                #+','  + str(((readadc_0(0) - NOwez) - (NOwe0/NOae0)*(readadc_0(1) + NOaez)) / NOs)
                #+','  + str(((readadc_0(2) - NO2wez) - (readadc_0(3) + NO2aez)) / NO2s)
                #+','  + str(((readadc_0(4) - OXwez) - (readadc_0(5) + OXaez)) / OXs)
                #+','  + str(((readadc_0(6) - COwez) - (readadc_0(7) + COaez)) / COs)
                + ',' + str(data[self.DUST_PM2_5_ATM])
                + ',' + str(data[self.DUST_PM10_0_ATM])
                #+ ',' + str(NO_ppb)
                
                +'\n')
        f.close()
        time.sleep(0.2)



# file.write(time.strftime('%Y/%m/%d,%H:%M:%S') + ',' + str(data[self.DUST_PM1_0_CF1]) + ',' + str(data[self.DUST_PM2_5_CF1]) + ',' + str(data[self.DUST_PM10_0_CF1]) + ',' + str(data[self.DUST_PM1_0_ATM]) + ',' + str(data[self.DUST_PM2_5_ATM])+ ',' + str(data[self.DUST_PM10_0_ATM])+','+'\n')


# UART / USB Serial : 'dmesg | grep ttyUSB'
USB0 = '/dev/ttyUSB0'
UART = '/dev/ttyAMA0'

# USE PORT
SERIAL_PORT = USB0

# Baud Rate
Speed = 9600

# example
if __name__ == '__main__':

    # serial setting
    ser = serial.Serial(SERIAL_PORT, Speed, timeout=1)

    dust = PMS7003()

    while True:

        ser.flushInput()
        buffer = ser.read(1024)

        if (dust.protocol_chk(buffer)):

            print("DATA read success")

            # print data
            dust.print_serial(buffer)

        else:

            print("DATA read fail...")

    ser.close()


