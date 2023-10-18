"""
일정시간동안 IMU 센서의 가속도, 자이로, 지자기 센서값을 읽어와서 그래프로 표현하는 코드
duration을 조정하여 일정 시간 동안 데이터를 읽어오는 것도 가능 (단위: 초)
interval을 조정하여 데이터를 읽어오는 속도를 조정할 수 있음
실제로는 연산속도의 문제로 동일한 hz로 데이터를 읽어오는것은 불가능
interval = 0.00285 실험적으로 얻은 200hz의 데이터를 읽어오는 속도
PC USB포트에 IMU를 연결하고 COM port를 지정해주면 실행
Data format: "%float(OrientPitch),%float(OrientYaw),%float(OrientRoll),
              %float(CorrectedGyroX),%float(CorrectedGyroY),%float(CorrectedGyroZ),
              %float(CorrectedAccelX),%float(CorrectedAccelY),%float(CorrectedAccelZ),
              %float(CorrectedMagX),%float(CorrectedMagY),%float(CorrectedMagZ)
Asyncio를 이용하여 정확한 200hz의 데이터를 읽어오도록 하려고 하였지만 
Asyncio의 특성상 최대 15미리 초만을 구현 가능함
threading의 경우 함수가 스레드 안에서 동작하지 않는 문제 발생
"""
from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *
import matplotlib.pyplot as plt
import time
import threading

class CorrectedComponentSensorData():

    def __init__(self, interval):
        com = USB_ExampleClass.UsbCom()
        print("com: ", com , type(com))
        # Create sensor instance. This will call the open function of the communication object.
        self.sensor = ThreeSpaceSensor(com)
        self.OrientPitch, self.OrientYaw, self.OrientRoll = [],[],[]
        self.acc_x, self.acc_y, self.acc_y = [],[],[]
        self.gyro_x, self.gyro_y, self.gyro_z = [],[],[]
        self.mag_x, self.mag_y, self.mag_z = [],[],[]
        self.interval = interval
        self.start_time = 0
        
        def read_imu(self,sensor):
            reading = sensor.getAllCorrectedComponentSensorData()
            # Save component values to lists
            self.OrientPitch.append(reading[0])
            self.OrientYaw.append(reading[1])
            self.OrientRoll.append(reading[2])
            self.acc_x.append(reading[3])
            self.acc_y.append(reading[4])
            self.acc_z.append(reading[5])
            self.gyro_x.append(reading[6])
            self.gyro_y.append(reading[7])
            self.gyro_z.append(reading[8])
            self.mag_x.append(reading[9])
            self.mag_y.append(reading[10])
            self.mag_z.append(reading[11])
            pass
        
        def start_read_imu(self):
            self.start_time = time.time()
            count = 0
            while True:
                if time.time() - self.start_time > self.interval * count:
                    read_imu(self, self.sensor)
                    count += 1
        
        def stop_imu(self):
            self.sensor.close()
            Orient = self.OrientPitch, self.OrientYaw, self.OrientRoll
            Accel = self.acc_x, self.acc_y, self.acc_z
            Gyro = self.gyro_x, self.gyro_y, self.gyro_z
            Mag = self.mag_x, self.mag_y, self.mag_z
            print("IMU data reading stopped")
            print("Working time: ", time.time() - self.start_time )
            return Orient, Accel, Gyro, Mag