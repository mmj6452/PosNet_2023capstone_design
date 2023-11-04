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
import pandas as pd
import datetime
import time

# Create communication object instance.
com = USB_ExampleClass.UsbCom()
print("com: ", com , type(com))
# Create sensor instance. This will call the open function of the communication object.
sensor = ThreeSpaceSensor(com)

# Initialize lists to store data
OrientPitch = []
OrientYaw = []
OrientRoll = []
acc_x = []
acc_y = []
acc_z = []
gyro_x = []
gyro_y = []
gyro_z = []
mag_x = []
mag_y = []
mag_z = []
Time = []

# Record data for 10 seconds with a 0.05-second interval
start_time = time.time()
duration = 10  # seconds
interval = 0.005  # seconds
end_time = time.time() + duration

def plot():
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(4, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    acc_max = 12
    gyro_max = 2.5
    mag_max = 0.8


    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Value')
    axs[0].set_title('OrientPitch OrientYaw OrientRoll')
    # Plot Gyro Y
    axs[0].plot(OrientYaw,'g')  
    # Plot Gyro Z
    axs[0].plot(OrientRoll,'b')


    # Plot Gyro X
    axs[1].plot(acc_x,'r')
    axs[1].set_ylim(-acc_max, acc_max)
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Value')
    axs[1].set_title('CorrectedGyroX CorrectedGyroY CorrectedGyroZ')
    # Plot Gyro Y
    axs[1].plot(acc_y,'g')
    # Plot Gyro Z
    axs[1].plot(acc_z,'b')


    # Plot Accel X
    axs[2].plot(gyro_x,'r')
    axs[2].set_ylim(-gyro_max, gyro_max)
    axs[2].set_title('CorrectedAccelX CorrectedAccelY CorrectedAccelZ')
    # Plot Accel Y
    axs[2].plot(gyro_y,'g')
    # Plot Accel Z
    axs[2].plot(gyro_z,'b')



    # Plot Mag X
    axs[3].plot(mag_x,'r')
    axs[3].set_ylim(-mag_max, mag_max)
    axs[3].set_xlabel('Sample')
    axs[3].set_ylabel('Value')
    axs[3].set_title('CorrectedMagX CorrectedMagY CorrectedMagZ')
    # Plot Mag Y
    axs[3].plot(mag_y,'g')
    # Plot Mag Z
    axs[3].plot(mag_z,'b')


    plt.tight_layout()
    plt.show()

def read_imu():
    reading = sensor.getAllRawComponentSensorData()

    # Save component values to lists
    #티임함수 시간을 년도, 월, 일, 시간, 분, 초로 나누어서 String으로 저장
    current_time  = time.time()
    time_struct = time.gmtime(current_time)
    #time_struct = time_struct.tm_hour, time_struct.tm_min, time_struct.tm_sec
    Time.append(time_struct)
    OrientPitch.append(reading[0])
    OrientYaw.append(reading[1])
    OrientRoll.append(reading[2])
    acc_x.append(reading[3])
    acc_y.append(reading[4])
    acc_z.append(reading[5])
    gyro_x.append(reading[6])
    gyro_y.append(reading[7])
    gyro_z.append(reading[8])
    mag_x.append(reading[9])
    mag_y.append(reading[10])
    mag_z.append(reading[11])


last_time = time.time()
count = 0
while time.time() < end_time:
    if time.time() - last_time > interval * count:
        print("loop: ", count)
        read_imu()
        count += 1

print(len(acc_x))
# Close the sensor
sensor.cleanup()
ending_time = time.time()
print("time: ", ending_time - start_time)

#t센서값을 CSV파일로 저장


data = {'Time': Time,
        'CorrectedGyroX': acc_x,
        'CorrectedGyroY': acc_y,
        'CorrectedGyroZ': acc_z,
        'CorrectedAccelX': gyro_x,
        'CorrectedAccelY': gyro_y,
        'CorrectedAccelZ': gyro_z,
        'CorrectedMagX': mag_x,
        'CorrectedMagY': mag_y,
        'CorrectedMagZ': mag_z}
#데이터를 주소에 저장
df = pd.DataFrame(data)
df.to_csv('PosNet_2023capstone_design/NSY/ThreeSpaceAPI_py3_beta_V0.6/Data/IMU_data.csv', index=False)
print("csv file saved")


plot()
