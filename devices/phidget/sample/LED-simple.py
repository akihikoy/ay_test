#!/usr/bin/env python

"""Copyright 2010 Phidgets Inc.
This work is licensed under the Creative Commons Attribution 2.5 Canada License. 
To view a copy of this license, visit http://creativecommons.org/licenses/by/2.5/ca/
"""

__author__ = 'Adam Stelmack'
__version__ = '2.1.8'
__date__ = 'May 17 2010'

#Basic imports
from ctypes import *
import sys
#Phidget specific imports
from Phidgets.PhidgetException import PhidgetErrorCodes, PhidgetException
from Phidgets.Events.Events import AttachEventArgs, DetachEventArgs, ErrorEventArgs
from Phidgets.Devices.LED import LED, LEDCurrentLimit, LEDVoltage
from time import sleep
from Phidgets.Phidget import PhidgetLogLevel


#Create an LED object
try:
    led = LED()
except RuntimeError as e:
    print("Runtime Exception: %s" % e.details)
    print("Exiting....")
    exit(1)

#Information Display Function
def displayDeviceInfo():
    print("|------------|----------------------------------|--------------|------------|")
    print("|- Attached -|-              Type              -|- Serial No. -|-  Version -|")
    print("|------------|----------------------------------|--------------|------------|")
    print("|- %8s -|- %30s -|- %10d -|- %8d -|" % (led.isAttached(), led.getDeviceName(), led.getSerialNum(), led.getDeviceVersion()))
    print("|------------|----------------------------------|--------------|------------|")

#Event Handler Callback Functions
def ledAttached(e):
    attached = e.device
    print("LED %i Attached!" % (attached.getSerialNum()))

def ledDetached(e):
    detached = e.device
    print("LED %i Detached!" % (detached.getSerialNum()))

def ledError(e):
    try:
        source = e.device
        print("LED %i: Phidget Error %i: %s" % (source.getSerialNum(), e.eCode, e.description))
    except PhidgetException as e:
        print("Phidget Exception %i: %s" % (e.code, e.details))

#Main Program Code
try:
	#logging example, uncomment to generate a log file
    #led.enableLogging(PhidgetLogLevel.PHIDGET_LOG_VERBOSE, "phidgetlog.log")

    led.setOnAttachHandler(ledAttached)
    led.setOnDetachHandler(ledDetached)
    led.setOnErrorhandler(ledError)
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))
    exit(1)

print("Opening phidget object...")

try:
    led.openPhidget()
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))
    exit(1)

print("Waiting for attach....")

try:
    led.waitForAttach(10000)
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))
    try:
        led.closePhidget()
    except PhidgetException as e:
        print("Phidget Exception %i: %s" % (e.code, e.details))
        exit(1)
    exit(1)
else:
    displayDeviceInfo()

print("Setting the output current limit and voltage levels to the default values....")
print("This is only supported on the 1031/1032 - LED Advanced")

#try to set these values, if we get an exception, it means most likely we are using an old 1030 LED board instead of a 1031 LED Advanced board
try:
    led.setCurrentLimit(LEDCurrentLimit.CURRENT_LIMIT_20mA)
    led.setVoltage(LEDVoltage.VOLTAGE_2_75V)
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))

print("Setting the indexed current limits....")
print("This is only supported on the 1032 - LED Advanced")

#try to set these values, if we get an exception, it means most likely we are using an old 1030 or 1031 LED board instead of a 1032 LED Advanced board
try:
	for i in range(64):
		led.setCurrentLimitIndexed(i, 40)
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))


print("Turning on LED's...")

for i in range(64):
    led.setBrightness(i, 100)
    sleep(0.1)

print("Turning off LED's...")

for i in range(64):
    led.setBrightness(i, 0)
    sleep(0.1)

print("Press Enter to quit....")

chr = sys.stdin.read(1)

print("Closing...")

try:
    led.closePhidget()
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))
    print("Exiting...")
    exit(1)

print("Done.")
exit(0)