"""
일정시간동안 IMU 센서의 가속도, 자이로, 지자기 센서값을 읽어와서 그래프로 표현하는 코드
duration을 조정하여 일정 시간 동안 데이터를 읽어오는 것도 가능 (단위: 초)
interval을 조정하여 데이터를 읽어오는 속도를 조정할 수 있음
PC USB포트에 IMU를 연결하고 COM port를 지정해주면 실행
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
duration = 10  # seconds
interval = 0.05  # seconds
end_time = time.time() + duration

while time.time() < end_time:
    # Get reading from sensor, returns as tuple.
    reading = sensor.getAllCorrectedComponentSensorData()

    # Save component values to lists
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

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

acc_max = 12
gyro_max = 2.5
mag_max = 0.8


# Plot Accel X
axs[0, 0].plot(acc_x)
axs[0, 0].set_ylim(-acc_max, acc_max)
axs[0, 0].set_xlabel('Sample')
axs[0, 0].set_ylabel('Value')
axs[0, 0].set_title('Accel X')

# Plot Accel Y
axs[0, 1].plot(acc_y)
axs[0, 1].set_ylim(-acc_max, acc_max)
axs[0, 1].set_xlabel('Sample')
axs[0, 1].set_ylabel('Value')
axs[0, 1].set_title('Accel Y')

# Plot Accel Z
axs[0, 2].plot(acc_z)
axs[0, 2].set_ylim(-acc_max, acc_max)
axs[0, 2].set_xlabel('Sample')
axs[0, 2].set_ylabel('Value')
axs[0, 2].set_title('Accel Z')

# Plot Gyro X
axs[1, 0].plot(gyro_x)
axs[1, 0].set_ylim(-gyro_max, gyro_max)
axs[1, 0].set_xlabel('Sample')
axs[1, 0].set_ylabel('Value')
axs[1, 0].set_title('Gyro X')

# Plot Gyro Y
axs[1, 1].plot(gyro_y)
axs[1, 1].set_ylim(-gyro_max, gyro_max)
axs[1, 1].set_xlabel('Sample')
axs[1, 1].set_ylabel('Value')
axs[1, 1].set_title('Gyro Y')

# Plot Gyro Z
axs[1, 2].plot(gyro_z)
axs[1, 2].set_ylim(-gyro_max, gyro_max)
axs[1, 2].set_xlabel('Sample')
axs[1, 2].set_ylabel('Value')
axs[1, 2].set_title('Gyro Z')

# Plot Mag X
axs[2, 0].plot(mag_x)
axs[2, 0].set_ylim(-mag_max, mag_max)
axs[2, 0].set_xlabel('Sample')
axs[2, 0].set_ylabel('Value')
axs[2, 0].set_title('Mag X')

# Plot Mag Y
axs[2, 1].plot(mag_y)
axs[2, 1].set_ylim(-mag_max, mag_max)
axs[2, 1].set_xlabel('Sample')
axs[2, 1].set_ylabel('Value')
axs[2, 1].set_title('Mag Y')

# Plot Mag Z
axs[2, 2].plot(mag_z)
axs[2, 2].set_ylim(-mag_max, mag_max)
axs[2, 2].set_xlabel('Sample')
axs[2, 2].set_ylabel('Value')
axs[2, 2].set_title('Mag Z')

plt.tight_layout()
plt.show()
