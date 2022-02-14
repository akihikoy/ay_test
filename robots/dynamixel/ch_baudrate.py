#!/usr/bin/python
#Change baud rate.
import argparse
parser= argparse.ArgumentParser(description='Change the baud rate of a Dynamixel.')
addarg= lambda *args,**kwargs: parser.add_argument(*args,**kwargs)
addarg('-dev', '--dev', metavar='DEV', dest='dev', default='/dev/ttyUSB0',
       help='Device location. default: /dev/ttyUSB0')
addarg('-type', '--type', metavar='DXL_TYPE', dest='dxl_type', default='XM430-W350',
       help='Dynamixel type. default: XM430-W350')
addarg('-id', '--id', metavar='DXL_ID', dest='dxl_id', default=1, type=int,
       help='Dynamixel ID. default: 1')
addarg('-cbr', '--curr_br', metavar='CURR_BAUDRATE', dest='curr_baudrate', default=57600, type=int,
       help='Current baud rate. default: 57600')
addarg('-nbr', '--new_br', metavar='NEW_BAUDRATE', dest='new_baudrate', default=2e6, type=int,
       help='New baud rate. default: 2e6')
args= parser.parse_args()
#print(args)

from dxl_util import *

#Setup the device
dxl= TDynamixel1(args.dxl_type, dev=args.dev)
dxl.Id= args.dxl_id
dxl.Baudrate= args.curr_baudrate
dxl.Setup()

dxl.Write('BAUD_RATE',dxl.BAUD_RATE.index(args.new_baudrate))

dxl.Quit()
