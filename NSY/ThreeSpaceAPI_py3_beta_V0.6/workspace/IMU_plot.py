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
"""

from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *
import matplotlib.pyplot as plt
import time

# Create communication object instance.
com = USB_ExampleClass.UsbCom()

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

# Record data for 10 seconds with a 0.05-second interval
duration = 3  # seconds
interval = 0.00285  # seconds
end_time = time.time() + duration

while time.time() < end_time:
    # Get reading from sensor, returns as tuple.
    reading = sensor.getAllRawComponentSensorData()

    # Save component values to lists
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

    time.sleep(interval)

# Close the sensor
sensor.cleanup()

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
print(len(acc_x))
