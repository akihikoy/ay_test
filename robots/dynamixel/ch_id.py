#!/usr/bin/python3
#Change ID.
import argparse
parser= argparse.ArgumentParser(description='Change the ID of a Dynamixel.')
addarg= lambda *args,**kwargs: parser.add_argument(*args,**kwargs)
addarg('-dev', '--dev', metavar='DEV', dest='dev', default='/dev/ttyUSB0',
       help='Device location. default: /dev/ttyUSB0')
addarg('-type', '--type', metavar='DXL_TYPE', dest='dxl_type', default='XM430-W350',
       help='Dynamixel type. default: XM430-W350')
addarg('-cid', '--curr_id', metavar='CURR_ID', dest='curr_id', default=1, type=int,
       help='Current Dynamixel ID. default: 1')
addarg('-nid', '--new_id', metavar='NEW_ID', dest='new_id', type=int, required=True,
       help='New Dynamixel ID.')
addarg('-br', metavar='BAUDRATE', dest='baudrate', default=2e6, type=int,
       help='Baud rate. default: 2e6')
args= parser.parse_args()
#print(args)

from dxl_util import *

#Setup the device
dxl= TDynamixel1(args.dxl_type, dev=args.dev)
dxl.Id= args.curr_id
dxl.Baudrate= args.baudrate
dxl.Setup()

#Change ID
dxl.Write('ID',args.new_id)

dxl.Quit()
