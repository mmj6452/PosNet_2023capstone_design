"""
CommandResponseWireless.py

There are two main ways of using Three Space Sensors: Command and Response, and
Streaming. This example shows how to use Command and Response to get samples
from a wireless sensor then log them to a file. This example uses the
USB_Example class but that is able to be swapped with any communication class
that works for your use case.

Setup:
Copy example file into same folder as ThreeSpaceAPI.py and USB_ExampleClass
Connect 3-Space Dongle to PC using usb cable.
Ensure that 3-Space Wireless Sensor serial number is set to logical ID 0 in
Dongle settings.
Power on 3-Space Wireless Sensor.

"""
from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *

# Create communication object instance.
com = USB_ExampleClass.UsbCom()
# Create sensor instance. This will call the open function of the communication object.
sensor = ThreeSpaceSensor(com)
# Create list to hold data
dataList = []
# Specify the logical ID that our sensor is located at
logicalID = 0
# Specify the amount of data points we want.
amountToLog = 250
print("Starting data collection!")
for i in range(amountToLog):
    # Get reading from sensor, returns as tuple, and includes header bytes
    reading = sensor.getTaredOrientation(logicalID=logicalID)
    # save reading to data list
    dataList.append(reading)
# close communication object and join any spawned threads
sensor.cleanup()
print("Logging data to file!")
# Create log file
logFile = open("exampleLog.txt","w")
for data in dataList:
    # Convert tuple to string, strips parenthesis, and adds new line before writing to file.
    logFile.write(str(data).strip('()') + '\n')

logFile.close()
print("Done!")