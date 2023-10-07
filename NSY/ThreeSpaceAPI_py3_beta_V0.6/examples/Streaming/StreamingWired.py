"""
StreamingWired.py

There are two main ways of using Three Space Sensors: Command and Response,
and Streaming. This example shows how to use Streaming to get samples from a
wired sensor then log them to a file. This example uses the USB_Example class
but that is able to be swapped with any communication class that works for your
use case.

Setup:
Copy example file into same folder as ThreeSpaceAPI.py and USB_ExampleClass
Connect 3-Space Sensor to PC using usb cable.
"""
import time

from exampleComClasses import USB_ExampleClass as comClass
# from ThreeSpaceAPI import *
from ExampleStaticAPI import *
# helper function
def hertzToInterval(hertz):
    return int(1000000/hertz)

# Create communication object instance.
com = comClass.UsbCom()
# Create sensor instance. This will call the open function of the communication object
# Set our buffer length to desired length.
sensor = ThreeSpaceSensor(com,streamingBufferLen=1000)
# Create log file
logFile = open("exampleLog.csv", "w+")
logFile.write("Success,Timestamp us,Command Echo,Data Length,Orientation X,Orientation Y,Orientation Z,Gyro X,Gyro Y,Gyro Z,Accel X,Accel Y,Accel Z,Mag X,Mag Y,Mag Z\n")

# Specify the amount of data points we want.
amountToLog = 2000

sensor.setResponseHeaderBitfield(3)
# Set sensor to stream Temperature and its orientation
sensor.setStreamingSlots(Streamable.READ_TARED_ORIENTATION_AS_EULER,Streamable.READ_ALL_CORRECTED_COMPONENT_SENSOR_DATA)
# Set sensor to stream at 100Hz until we tell it to stop with no start delay
# All arguments are in microseconds
sensor.setStreamingTiming(hertzToInterval(100),STREAM_CONTINUOUSLY,0)
print(sensor.getSerialNumber())

print("Starting Streaming!")
start = time.time()
sensor.startStreaming()
data = None
dataPoints = 0
percentage = 0
while dataPoints < amountToLog:
    # Streamed data is added to buffer. This method safely accesses that buffer
    data = sensor.getOldestStreamingPacket()
    # getOldestStreamingPacket() will return None if there is no data in buffer
    if data is not None:
        # Convert tuple to string, strips parenthesis, and adds new line before writing to file.
        logFile.write(str(data).strip('()') + '\n')
        dataPoints += 1
        # Update and print percentage dots
        curPercentage = int((dataPoints/amountToLog)*100)
        if curPercentage > percentage:
            diff = curPercentage-percentage
            for i in range(1,diff+1):
                if (percentage + i) %10 == 0:
                    print("{}%".format(percentage+i))
                else:
                    print(". ",end='')
            percentage = curPercentage
    else:
        # allow the streaming thread time to fill the streaming buffer
        time.sleep(0)
stop = time.time()

sensor.stopStreaming()
print(f"time to stream {stop - start}")
# close communication object and join any spawned threads
sensor.cleanup()

logFile.close()
print("Done!")