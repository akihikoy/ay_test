#================================================================
#================================================================
# API-DIO(LNX)
# Input and Output Sample
#                                                CONTEC Co., Ltd.
#================================================================
#================================================================

import ctypes
import sys
import cdio

#================================================================
# Command Define
#================================================================
COMMAND_ERROR = 0      # Error
COMMAND_INP_PORT = 1   # 1 Port Input
COMMAND_INP_BIT = 2    # 1 Bit Input
COMMAND_OUT_PORT = 3   # 1 Port Output
COMMAND_OUT_BIT = 4    # 1 Bit Output
COMMAND_ECHO_PORT = 5  # 1 Port Echo Back
COMMAND_ECHO_BIT = 6   # 1 Bit Echo Back
COMMAND_QUIT = 7       # End


#================================================================
# Function that checks if a string can be converted to a number
#================================================================
def isnum(str, base):
    try:
        if 16 == base:
            int(str, 16)
        else:
            int(str)
    except:
        return False
    return True


#================================================================
# Main Function
#================================================================
def main():
    dio_id = ctypes.c_short()
    io_data = ctypes.c_ubyte()
    port_no = ctypes.c_short()
    bit_no = ctypes.c_short()
    err_str = ctypes.create_string_buffer(256)

    #----------------------------------------
    # Initialization
    #----------------------------------------
    dev_name = input('Input device name: ')
    lret = cdio.DioInit(dev_name.encode(), ctypes.byref(dio_id))
    if lret != cdio.DIO_ERR_SUCCESS:
        cdio.DioGetErrorString(lret, err_str)
        print(f"DioInit = {lret}: {err_str.value.decode('utf-8')}")
        sys.exit()
    #----------------------------------------
    # Loop that Wait Input
    #----------------------------------------
    while True:
        #----------------------------------------
        # Display Command
        #----------------------------------------
        print('')
        print('--------------------')
        print(' Menu')
        print('--------------------')
        print('ip : port input')
        print('ib : bit  input')
        print('op : port output')
        print('ob : bit  output')
        print('ep : port echoback')
        print('eb : bit  echoback')
        print('q  : quit')
        print('--------------------')
        buf = input('input command: ')
        #----------------------------------------
        # Distinguish Command
        #----------------------------------------
        command = COMMAND_ERROR
        #----------------------------------------
        # 1 Port Input
        #----------------------------------------
        if buf == 'ip':
            command = COMMAND_INP_PORT
        #----------------------------------------
        # 1 Bit Input
        #----------------------------------------
        if buf == 'ib':
            command = COMMAND_INP_BIT
        #----------------------------------------
        # 1 Port Output
        #----------------------------------------
        if buf == 'op':
            command = COMMAND_OUT_PORT
        #----------------------------------------
        # 1 Bit Output
        #----------------------------------------
        if buf == 'ob':
            command = COMMAND_OUT_BIT
        #----------------------------------------
        # 1 Port Echo Back
        #----------------------------------------
        if buf == 'ep':
            command = COMMAND_ECHO_PORT
        #----------------------------------------
        # 1 Bit Echo Back
        #----------------------------------------
        if buf == 'eb':
            command = COMMAND_ECHO_BIT
        #----------------------------------------
        # End
        #----------------------------------------
        if buf == 'q':
            command = COMMAND_QUIT
        #----------------------------------------
        # Input Port Number and Bit Number
        #----------------------------------------
        if(command == COMMAND_INP_PORT or
           command == COMMAND_OUT_PORT or
           command == COMMAND_ECHO_PORT):
           while True:
                buf = input('input port number: ')
                if False == isnum(buf, 10):
                   continue
                port_no = ctypes.c_short(int(buf))
                break
        elif(command == COMMAND_INP_BIT or
             command == COMMAND_OUT_BIT or
             command == COMMAND_ECHO_BIT):
             while True:
                buf = input('input bit number: ')
                if False == isnum(buf, 10):
                   continue
                bit_no = ctypes.c_short(int(buf))
                break
        #----------------------------------------
        # Input Data
        #----------------------------------------
        if(command == COMMAND_OUT_PORT or
           command == COMMAND_OUT_BIT):
           while True:
                buf = input('input data (hex): ')
                if False == isnum(buf, 16):
                   continue
                io_data = ctypes.c_ubyte(int(buf, 16))
                break
        #----------------------------------------
        # Execute Command and Display Result
        #----------------------------------------
        #----------------------------------------
        # 1 Port Input
        #----------------------------------------
        if command == COMMAND_INP_PORT:
            lret = cdio.DioInpByte(dio_id, port_no, ctypes.byref(io_data))
            if lret == cdio.DIO_ERR_SUCCESS:
                cdio.DioGetErrorString(lret, err_str)
                print(f'DioInpByte port = {port_no.value}: data = 0x{io_data.value:02x}')
            else:
                cdio.DioGetErrorString(lret, err_str)
                print(f"DioInpByte = {lret}: {err_str.value.decode('utf-8')}")
        #----------------------------------------
        # 1 Bit Input
        #----------------------------------------
        elif command == COMMAND_INP_BIT:
            lret = cdio.DioInpBit(dio_id, bit_no, ctypes.byref(io_data))
            if lret == cdio.DIO_ERR_SUCCESS:
                cdio.DioGetErrorString(lret, err_str)
                print(f'DioInpBit port = {bit_no.value}: data = 0x{io_data.value:02x}')
            else:
                cdio.DioGetErrorString(lret, err_str)
                print(f"DioInpBit = {lret}: {err_str.value.decode('utf-8')}")
        #----------------------------------------
        # 1 Port Output
        #----------------------------------------
        elif command == COMMAND_OUT_PORT:
            lret = cdio.DioOutByte(dio_id, port_no, io_data)
            if lret == cdio.DIO_ERR_SUCCESS:
                cdio.DioGetErrorString(lret, err_str)
                print(f'DioOutByte port = {port_no.value}: data = 0x{io_data.value:02x}')
            else:
                cdio.DioGetErrorString(lret, err_str)
                print(f"DioOutByte = {lret}: {err_str.value.decode('utf-8')}")
        #----------------------------------------
        # 1 Bit Output
        #----------------------------------------
        elif command == COMMAND_OUT_BIT:
            lret = cdio.DioOutBit(dio_id, bit_no, io_data)
            if lret == cdio.DIO_ERR_SUCCESS:
                cdio.DioGetErrorString(lret, err_str)
                print(f'DioOutBit port = {bit_no.value}: data = 0x{io_data.value:02x}')
            else:
                cdio.DioGetErrorString(lret, err_str)
                print(f"DioOutBit = {lret}: {err_str.value.decode('utf-8')}")
        #----------------------------------------
        # 1 Port Echo Back
        #----------------------------------------
        elif command == COMMAND_ECHO_PORT:
            lret = cdio.DioEchoBackByte(dio_id, port_no, ctypes.byref(io_data))
            if lret == cdio.DIO_ERR_SUCCESS:
                cdio.DioGetErrorString(lret, err_str)
                print(f'DioEchoBackByte port = {port_no.value}: data = 0x{io_data.value:02x}')
            else:
                cdio.DioGetErrorString(lret, err_str)
                print(f"DioEchoBackByte = {lret}: {err_str.value.decode('utf-8')}")
        #----------------------------------------
        # 1 Bit Echo Back
        #----------------------------------------
        elif command == COMMAND_ECHO_BIT:
            lret = cdio.DioEchoBackBit(dio_id, bit_no, ctypes.byref(io_data))
            if lret == cdio.DIO_ERR_SUCCESS:
                cdio.DioGetErrorString(lret, err_str)
                print(f'DioEchoBackBit port = {bit_no.value}: data = 0x{io_data.value:02x}')
            else:
                cdio.DioGetErrorString(lret, err_str)
                print(f"DioEchoBackBit = {lret}: {err_str.value.decode('utf-8')}")
        #----------------------------------------
        # End
        #----------------------------------------
        elif command == COMMAND_QUIT:
            print(f'quit.')
            break
        #----------------------------------------
        # Error
        #----------------------------------------
        elif command == COMMAND_ERROR:
            print(f'error: {buf}.')
            break
    #----------------------------------------
    # Exit
    #----------------------------------------
    lret = cdio.DioExit(dio_id)
    if lret != cdio.DIO_ERR_SUCCESS:
        cdio.DioGetErrorString(lret, err_str)
        print(f"DioExit = {lret}: {err_str.value.decode('utf-8')}")
    #----------------------------------------
    # Terminate application
    #----------------------------------------
    sys.exit()


#----------------------------------------
# Call main function
#----------------------------------------
if __name__ == "__main__":
    main()
