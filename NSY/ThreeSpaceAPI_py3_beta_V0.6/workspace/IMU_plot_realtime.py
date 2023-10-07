"""
IMU데이터를 실시간으로 그래프로 표현하는 코드
Plot속도의 문제로 빠른 속도로 데이터를 읽어오는것은 불가능
테스팅 용도의 코드
PC USB포트에 IMU를 연결하고 COM port를 지정해주면 실행
"""

from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Create communication object instance.
com = USB_ExampleClass.UsbCom()

# Create sensor instance. This will call the open function of the communication object.
sensor = ThreeSpaceSensor(com)

# Create figure windows outside the loop
fig_accel = plt.figure()
fig_gyro = plt.figure()
fig_mag = plt.figure()

# Create subplots for each figure
ax_accel_x = fig_accel.add_subplot(311)
ax_accel_y = fig_accel.add_subplot(312)
ax_accel_z = fig_accel.add_subplot(313)

ax_gyro_x = fig_gyro.add_subplot(311)
ax_gyro_y = fig_gyro.add_subplot(312)
ax_gyro_z = fig_gyro.add_subplot(313)

ax_mag_x = fig_mag.add_subplot(311)
ax_mag_y = fig_mag.add_subplot(312)
ax_mag_z = fig_mag.add_subplot(313)

while True:
    # Get reading from sensor, returns as tuple.
    reading = sensor.getAllCorrectedComponentSensorData()
    print(reading)
    print("")

    accelData = reading[3:6]
    gyroData = reading[6:9]
    magData = reading[9:12]

    # Clear the previous plots
    ax_accel_x.cla()
    ax_accel_y.cla()
    ax_accel_z.cla()
    ax_gyro_x.cla()
    ax_gyro_y.cla()
    ax_gyro_z.cla()
    ax_mag_x.cla()
    ax_mag_y.cla()
    ax_mag_z.cla()

    # Set Y-axis limits for each subplot
    ax_accel_x.set_ylim(-2, 2)
    ax_accel_y.set_ylim(-2, 2)
    ax_accel_z.set_ylim(-2, 2)
    ax_gyro_x.set_ylim(-2, 2)
    ax_gyro_y.set_ylim(-2, 2)
    ax_gyro_z.set_ylim(-2, 2)
    ax_mag_x.set_ylim(-2, 2)
    ax_mag_y.set_ylim(-2, 2)
    ax_mag_z.set_ylim(-2, 2)

    # Plotting in each subplot
    ax_accel_x.plot([0], accelData[0], 'bo-')
    ax_accel_x.set_title('Accel Data X')
    ax_accel_x.set_ylabel('Value')

    ax_accel_y.plot([1], accelData[1], 'go-')
    ax_accel_y.set_title('Accel Data Y')
    ax_accel_y.set_ylabel('Value')

    ax_accel_z.plot([2], accelData[2], 'ro-')
    ax_accel_z.set_title('Accel Data Z')
    ax_accel_z.set_ylabel('Value')

    ax_gyro_x.plot([0], gyroData[0], 'bo-')
    ax_gyro_x.set_title('Gyro Data X')
    ax_gyro_x.set_ylabel('Value')

    ax_gyro_y.plot([1], gyroData[1], 'go-')
    ax_gyro_y.set_title('Gyro Data Y')
    ax_gyro_y.set_ylabel('Value')

    ax_gyro_z.plot([2], gyroData[2], 'ro-')
    ax_gyro_z.set_title('Gyro Data Z')
    ax_gyro_z.set_ylabel('Value')

    ax_mag_x.plot([0], magData[0], 'bo-')
    ax_mag_x.set_title('Mag Data X')
    ax_mag_x.set_ylabel('Value')

    ax_mag_y.plot([1], magData[1], 'go-')
    ax_mag_y.set_title('Mag Data Y')
    ax_mag_y.set_ylabel('Value')

    ax_mag_z.plot([2], magData[2], 'ro-')
    ax_mag_z.set_title('Mag Data Z')
    ax_mag_z.set_ylabel('Value')

    # Adjust layout and display
    fig_accel.tight_layout()
    fig_gyro.tight_layout()
    fig_mag.tight_layout()
    plt.pause(0.005)

# close communication object and join any spawned threads
sensor.cleanup()
