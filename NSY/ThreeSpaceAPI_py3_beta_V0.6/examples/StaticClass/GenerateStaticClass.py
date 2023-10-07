"""
GenerateStaticClass.py

The default setup of the ThreeSpaceAPI is to detect the type of sensor then
dynamically generate the functions that that type of sensor supports.
This is to make maintenance of the API easier and also displays some of the
great features of Python. Since we do generate the code and create the
definitions for the functions when we connect to a sensor it mightbe slower then
a production environment would prefer, to solve this we created the
generateStaticClass method this method will write all the generated functions
and anything else that is necessary to a new python file that will then be
usable as an API for that sensor type. This can also be helpful for debugging if
you are getting unexpected behavior or errors.

In this Example we show you the process for making a static class that we will
demonstrate using in the UsingStaticClass.py example.

Setup:
Copy example file into same folder as ThreeSpaceAPI.py and USB_ExampleClass
Connect 3-Space Sensor to PC using usb cable.
"""
from exampleComClasses import USB_ExampleClass
from ThreeSpaceAPI import *

# Create communication object instance.
com = USB_ExampleClass.UsbCom()
# Create sensor instance. This will call the open function of the communication object.
# All methods will be generated on this instance creation.
sensor = ThreeSpaceSensor(com)
# By default this method will name the new API based on your sensor type but you can give it a filename as well.
# All methods generated on instance creation will be written to this file.
sensor.generateStaticClass(filename="../../ExampleStaticAPI.py")
# close communication object and join any spawned threads
sensor.cleanup()
print("Done!")
