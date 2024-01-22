#================================================================
# cdio.py
# module file for CONTEC Digital I/O device
#                                                CONTEC.Co., Ltd.
#================================================================
import ctypes

cdio_dll = ctypes.cdll.LoadLibrary('libcdio.so')

#----------------------------------------
# Macro definition
#----------------------------------------
#----------------------------------------
# Message
#----------------------------------------
DIOM_INTERRUPT = 0x1300
DIOM_TRIGGER = 0x1340
DIO_DMM_STOP = 0x1400
DIO_DMM_COUNT = 0x1440
#----------------------------------------
# Error code
#----------------------------------------
DIO_ERR_SUCCESS = 0                             # Normal complete
DIO_ERR_INI_RESOURCE = 1                        # Failed in the acquisition of resource.
DIO_ERR_INI_INTERRUPT = 2                       # Failed in the registration of interrupt routine.
DIO_ERR_INI_MEMORY = 3                          # Failed in the memory allocation. This error hardly occurs. If this error has occurred, please install more memory.
DIO_ERR_INI_REGISTRY = 4                        # Failed in accessing the setting file.
DIO_ERR_DLL_DEVICE_NAME = 10000                 # Device name which isn't registered in setting file is specified.
DIO_ERR_DLL_INVALID_ID = 10001                  # Invalid ID is specified. Make sure whether initialization function normally complete. And, make sure the scope of variable in which ID is stored.
DIO_ERR_DLL_CALL_DRIVER = 10002                 # Driver cannot be called (failed in ioctl).
DIO_ERR_DLL_CREATE_FILE = 10003                 # Failed in creating file (open failed).
DIO_ERR_DLL_CLOSE_FILE = 10004                  # Failed in closing file (close failed).
DIO_ERR_DLL_CREATE_THREAD = 10005               # Failed in creating thread.
DIO_ERR_INFO_INVALID_DEVICE = 10050             # Specified device name isn't found. Please check the spelling.
DIO_ERR_INFO_NOT_FIND_DEVICE = 10051            # The usable device isn't found.
DIO_ERR_INFO_INVALID_INFOTYPE = 10052           # The specified device information type is outside the range.
DIO_ERR_DLL_BUFF_ADDRESS = 10100                # Invalid data buffer address.
DIO_ERR_DLL_TRG_KIND = 10300                    # Trigger type is outside the range.
DIO_ERR_DLL_CALLBACK = 10400                    # Invalid address of callback function.
DIO_ERR_DLL_DIRECTION = 10500                   # I/O direction is outside the setting range.
DIO_ERR_SYS_MEMORY = 20000                      # Failed in memory. This error hardly occurs. If this error has occurred, please install more memory.
DIO_ERR_SYS_NOT_SUPPORTED = 20001               # This function cannot be used in this device.
DIO_ERR_SYS_BOARD_EXECUTING = 20002             # It cannot perform because the device is executing.
DIO_ERR_SYS_USING_OTHER_PROCESS = 20003         # It cannot perform because the other process is using the device.
DIO_ERR_SYS_NOT_SUPPORT_KERNEL = 20004          # It is not supporting in the kernel of this version.
DIO_ERR_SYS_PORT_NO = 20100                     # PortNo is outside the setting range.
DIO_ERR_SYS_PORT_NUM = 20101                    # Number of ports is outside the setting range.
DIO_ERR_SYS_BIT_NO = 20102                      # BitNo is outside the range.
DIO_ERR_SYS_BIT_NUM = 20103                     # Number of bits is outside the setting range.
DIO_ERR_SYS_BIT_DATA = 20104                    # Bit data is not 0 or 1.
DIO_ERR_SYS_INT_BIT = 20200                     # Interrupt bit is outside the setting range.
DIO_ERR_SYS_INT_LOGIC = 20201                   # Interrupt logic is outside the setting range.
DIO_ERR_SYS_TRG_LOGIC = 20202                   # Trigger logic is outside the setting range.
DIO_ERR_SYS_TIM = 20300                         # Timer value is outside the setting range. Error in trigger function.
DIO_ERR_SYS_FILTER = 20400                      # Filter value is outside the setting range.
DIO_ERR_SYS_8255 = 20500                        # 8255 chip number is output the setting range
DIO_ERR_SYS_DIRECTION = 50000                   # I/O direction is outside the setting range.
DIO_ERR_SYS_SIGNAL = 50001                      # Usable signal is outside the setting range.
DIO_ERR_SYS_START = 50002                       # Usable start conditions are outside the setting range.
DIO_ERR_SYS_CLOCK = 50003                       # Clock conditions are outside the setting range.
DIO_ERR_SYS_CLOCK_VAL = 50004                   # Clock value is outside the setting range.
DIO_ERR_SYS_CLOCK_UNIT = 50005                  # Clock value unit is outside the setting range.
DIO_ERR_SYS_STOP = 50006                        # Stop conditions are outside the setting range.
DIO_ERR_SYS_STOP_NUM = 50007                    # Stop number is outside the setting range.
DIO_ERR_SYS_RESET = 50008                       # Contents of reset are outside the setting range.
DIO_ERR_SYS_LEN = 50009                         # Data number is outside the setting range.
DIO_ERR_SYS_RING = 50010                        # Buffer repetition use setup is outside the setting range.
DIO_ERR_SYS_COUNT = 50011                       # Data transmission number is outside the setting range.
DIO_ERR_DM_BUFFER = 50100                       # Buffer was too large and has not secured.
DIO_ERR_DM_LOCK_MEMORY = 50101                  # Memory has not been locked.
DIO_ERR_DM_PARAM = 50102                        # Parameter error
DIO_ERR_DM_SEQUENCE = 50103                     # Procedure error of execution
#----------------------------------------
# Information type
#----------------------------------------
IDIO_DEVICE_TYPE = 0                # Device type(short)
IDIO_NUMBER_OF_8255 = 1             # 8255 number(int)
IDIO_IS_8255_BOARD = 2              # 8255 type(int)
IDIO_NUMBER_OF_DI_BIT = 3           # DI BIT(short)
IDIO_NUMBER_OF_DO_BIT = 4           # DO BIT(short)
IDIO_NUMBER_OF_DI_PORT = 5          # DI PORT(short)
IDIO_NUMBER_OF_DO_PORT = 6          # DO PORT(short)
IDIO_IS_POSITIVE_LOGIC = 7          # Positive logic(int)
IDIO_IS_ECHO_BACK = 8               # Echo back(int)
IDIO_IS_DIRECTION = 9               # DioSetIoDirection function available(int)
IDIO_IS_FILTER = 10                 # Digital filter available(int)
IDIO_NUMBER_OF_INT_BIT = 11         # Interruptable number of bits(short)
#----------------------------------------
# Interrupt, trigger rising, falling
#----------------------------------------
DIO_INT_NONE = 0                    # Interrupt:mask
DIO_INT_RISE = 1                    # Interrupt:rising
DIO_INT_FALL = 2                    # Interrupt:falling
DIO_TRG_NONE = 0                    # Trigger:mask
DIO_TRG_RISE = 1                    # Trigger:rising
DIO_TRG_FALL = 2                    # Trigger:falling
#----------------------------------------
# Device type
#----------------------------------------
DEVICE_TYPE_ISA = 0                 # ISA or C bus
DEVICE_TYPE_PCI = 1                 # PCI bus
DEVICE_TYPE_PCMCIA = 2              # PCMCIA
DEVICE_TYPE_USB = 3                 # USB
DEVICE_TYPE_FIT = 4                 # FIT
#----------------------------------------
# Direction
#----------------------------------------
PI_32 = 1                           # 32-bit input
PO_32 = 2                           # 32-bit output
PIO_1616 = 3                        # 16-bit input, 16-bit output
DIODM_DIR_IN = 0x1                  # Input
DIODM_DIR_OUT = 0x2                 # Output
#----------------------------------------
# Start
#----------------------------------------
DIODM_START_SOFT = 1                # Software start
DIODM_START_EXT_RISE = 2            # External trigger rising
DIODM_START_EXT_FALL = 3            # External trigger falling
DIODM_START_PATTERN = 4             # Patter matching(only input)
DIODM_START_EXTSIG_1 = 5            # SC connector EXTSIG1
DIODM_START_EXTSIG_2 = 6            # SC connector EXTSIG2
DIODM_START_EXTSIG_3 = 7            # SC connector EXTSIG3
#----------------------------------------
# Clock
#----------------------------------------
DIODM_CLK_CLOCK = 1                 # Internal clock(timer)
DIODM_CLK_EXT_TRG = 2               # External trigger
DIODM_CLK_HANDSHAKE = 3             # Hand shake
DIODM_CLK_EXTSIG_1 = 4              # SC connector EXTSIG1
DIODM_CLK_EXTSIG_2 = 5              # SC connector EXTSIG2
DIODM_CLK_EXTSIG_3 = 6              # SC connector EXTSIG3
#----------------------------------------
# Internal Clock
#----------------------------------------
DIODM_TIM_UNIT_S = 1                # 1s unit
DIODM_TIM_UNIT_MS = 2               # 1ms unit
DIODM_TIM_UNIT_US = 3               # 1us unit
DIODM_TIM_UNIT_NS = 4               # 1ns unit
#----------------------------------------
# Stop
#----------------------------------------
DIODM_STOP_SOFT = 1                 # Software stop
DIODM_STOP_EXT_RISE = 2             # External trigger rising
DIODM_STOP_EXT_FALL = 3             # External trigger falling
DIODM_STOP_NUM = 4                  # Stop transfer on specified transfer number
DIODM_STOP_EXTSIG_1 = 5             # SC connector EXTSIG1
DIODM_STOP_EXTSIG_2 = 6             # SC connector EXTSIG2
DIODM_STOP_EXTSIG_3 = 7             # SC connector EXTSIG3
#----------------------------------------
# ExtSig
#----------------------------------------
DIODM_EXT_START_SOFT_IN = 1         # Software start(pattern input)
DIODM_EXT_STOP_SOFT_IN = 2          # Software stop(pattern input)
DIODM_EXT_CLOCK_IN = 3              # Internal clock(pattern input)
DIODM_EXT_EXT_TRG_IN = 4            # External clock(pattern input)
DIODM_EXT_START_EXT_RISE_IN = 5     # External start trigger rising(pattern input)
DIODM_EXT_START_EXT_FALL_IN = 6     # External start trigger falling(pattern input)
DIODM_EXT_START_PATTERN_IN = 7      # Pattern matching(pattern input)
DIODM_EXT_STOP_EXT_RISE_IN = 8      # External stop trigger rising(pattern input)
DIODM_EXT_STOP_EXT_FALL_IN = 9      # External stop trigger falling(pattern input)
DIODM_EXT_CLOCK_ERROR_IN = 10       # Clock error(pattern input)
DIODM_EXT_HANDSHAKE_IN = 11         # Hand shake(pattern input)
DIODM_EXT_TRNSNUM_IN = 12           # Stop transfer on specified transfer number(pattern input)
	
DIODM_EXT_START_SOFT_OUT = 101      # Software start(pattern output)
DIODM_EXT_STOP_SOFT_OUT = 102       # Software stop(pattern output)
DIODM_EXT_CLOCK_OUT = 103           # Internal clock(pattern output)
DIODM_EXT_EXT_TRG_OUT = 104         # External clock(pattern output)
DIODM_EXT_START_EXT_RISE_OUT = 105  # External start trigger rising(pattern output)
DIODM_EXT_START_EXT_FALL_OUT = 106  # External start trigger falling(pattern output)
DIODM_EXT_STOP_EXT_RISE_OUT = 107   # External stop trigger rising(pattern output)
DIODM_EXT_STOP_EXT_FALL_OUT = 108   # External stop trigger falling(pattern output)
DIODM_EXT_CLOCK_ERROR_OUT = 109     # Clock error(pattern output)
DIODM_EXT_HANDSHAKE_OUT = 110       # Hand shake(pattern output)
#----------------------------------------
# Status
#----------------------------------------
DIODM_STATUS_BMSTOP = 0x1           # Indicates that bus master transfer is complete.
DIODM_STATUS_PIOSTART = 0x2         # Indicates that PIO input/output has started.
DIODM_STATUS_PIOSTOP = 0x4          # Indicates that PIO input/ouput has stopped.
DIODM_STATUS_TRGIN = 0x8            # Indicates that the start signal is inserted by external start.
DIODM_STATUS_OVERRUN = 0x10         # Indicates that start signals are inserted twice or more by external start.
#----------------------------------------
# Error
#----------------------------------------
DIODM_STATUS_FIFOEMPTY = 0x1        # Indicates that FIFO becomes vacant.
DIODM_STATUS_FIFOFULL = 0x2         # Indicates that FIFO is full due to input.
DIODM_STATUS_SGOVERIN = 0x4         # Indicates that the buffer has overflowed.
DIODM_STATUS_TRGERR = 0x8           # Indicates that start and stop signals have been inserted simultaneously by external start.
DIODM_STATUS_CLKERR = 0x10          # Indicates thar the next clock is inserted during data input/output by the external clock.
DIODM_STATUS_SLAVEHALT = 0x20
DIODM_STATUS_MASTERHALT = 0x40
#----------------------------------------
# Reset
#----------------------------------------
DIODM_RESET_FIFO_IN = 0x02          # Reset input FIFO
DIODM_RESET_FIFO_OUT = 0x04         # Reset output FIFO
#----------------------------------------
# Buffer Ring
#----------------------------------------
DIODM_WRITE_ONCE = 0                # Single transfer
DIODM_WRITE_RING = 1                # Unlimited transfer


#----------------------------------------
# Form of callback function
#----------------------------------------
PDIO_INT_CALLBACK = ctypes.CFUNCTYPE(None,
                                       ctypes.c_short, ctypes.c_short,
                                       ctypes.c_long, ctypes.c_long, ctypes.c_void_p)
PDIO_TRG_CALLBACK = ctypes.CFUNCTYPE(None,
                                       ctypes.c_short, ctypes.c_short,
                                       ctypes.c_long, ctypes.c_long, ctypes.c_void_p)
PDIO_STOP_CALLBACK = ctypes.CFUNCTYPE(None,
                                        ctypes.c_short, ctypes.c_short,
                                        ctypes.c_long, ctypes.c_void_p)
PDIO_COUNT_CALLBACK = ctypes.CFUNCTYPE(None,
                                        ctypes.c_short, ctypes.c_short,
                                        ctypes.c_long, ctypes.c_void_p)


#----------------------------------------
# Prototype definition
#----------------------------------------
#----------------------------------------
# Common function
#----------------------------------------
# C Prototype: long DioInit(char *device_name, short *id);
DioInit = cdio_dll.DioInit
DioInit.restype = ctypes.c_long
DioInit.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_short)]

# C Prototype: long DioExit(short id);
DioExit = cdio_dll.DioExit
DioExit.restype = ctypes.c_long
DioExit.argtypes = [ctypes.c_short]

# C Prototype: long DioResetDevice(short id);
DioResetDevice = cdio_dll.DioResetDevice
DioResetDevice.restype = ctypes.c_long
DioResetDevice.argtypes = [ctypes.c_short]

# C Prototype: long DioGetErrorString(long err_code, char *err_string);
DioGetErrorString = cdio_dll.DioGetErrorString
DioGetErrorString.restype = ctypes.c_long
DioGetErrorString.argtypes = [ctypes.c_long, ctypes.c_char_p]

#----------------------------------------
# Digital filter function
#----------------------------------------
# C Prototype: long DioSetDigitalFilter(short id, short filter_value);
DioSetDigitalFilter = cdio_dll.DioSetDigitalFilter
DioSetDigitalFilter.restype = ctypes.c_long
DioSetDigitalFilter.argtypes = [ctypes.c_short, ctypes.c_short]

# C Prototype: long DioGetDigitalFilter(short id, short *filter_value);
DioGetDigitalFilter = cdio_dll.DioGetDigitalFilter
DioGetDigitalFilter.restype = ctypes.c_long
DioGetDigitalFilter.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short)]

#----------------------------------------
# I/O Direction function
#----------------------------------------
# C Prototype: long DioSetIoDirection(short id, long dir);
DioSetIoDirection = cdio_dll.DioSetIoDirection
DioSetIoDirection.restype = ctypes.c_long
DioSetIoDirection.argtypes = [ctypes.c_short, ctypes.c_long]

# C Prototype: long DioGetIoDirection(short id, long *dir);
DioGetIoDirection = cdio_dll.DioGetIoDirection
DioGetIoDirection.restype = ctypes.c_long
DioGetIoDirection.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_long)]

# C Prototype: long DioSet8255Mode(short id, unsigned short chip_no, unsigned short ctrl_word);
DioSet8255Mode = cdio_dll.DioSet8255Mode
DioSet8255Mode.restype = ctypes.c_long
DioSet8255Mode.argtypes = [ctypes.c_short, ctypes.c_ushort, ctypes.c_ushort]

# C Prototype: long DioGet8255Mode(short id, unsigned short chip_no, unsigned short *ctrl_word);
DioGet8255Mode = cdio_dll.DioGet8255Mode
DioGet8255Mode.restype = ctypes.c_long
DioGet8255Mode.argtypes = [ctypes.c_short, ctypes.c_ushort, ctypes.POINTER(ctypes.c_ushort)]

#----------------------------------------
# Simple I/O function
#----------------------------------------
# C Prototype: long DioInpByte(short id, short port_no, unsigned char *data);
DioInpByte = cdio_dll.DioInpByte
DioInpByte.restype = ctypes.c_long
DioInpByte.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioInpBit(short id, short bit_no, unsigned char *data);
DioInpBit = cdio_dll.DioInpBit
DioInpBit.restype = ctypes.c_long
DioInpBit.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioInpByteSR(short id, short port_no, unsigned char *data, unsigned short *time_stamp, unsigned char mode);
DioInpByteSR = cdio_dll.DioInpByteSR
DioInpByteSR.restype = ctypes.c_long
DioInpByteSR.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ushort), ctypes.c_ubyte]

# C Prototype: long DioInpBitSR(short id, short bit_no, unsigned char *data, unsigned short *time_stamp, unsigned char mode);
DioInpBitSR = cdio_dll.DioInpBitSR
DioInpBitSR.restype = ctypes.c_long
DioInpBitSR.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ushort), ctypes.c_ubyte]

# C Prototype: long DioOutByte(short id, short port_no, unsigned char data);
DioOutByte = cdio_dll.DioOutByte
DioOutByte.restype = ctypes.c_long
DioOutByte.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_ubyte]

# C Prototype: long DioOutBit(short id, short bit_no, unsigned char data);
DioOutBit = cdio_dll.DioOutBit
DioOutBit.restype = ctypes.c_long
DioOutBit.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_ubyte]

# C Prototype: long DioEchoBackByte(short id, short port_no, unsigned char *data);
DioEchoBackByte = cdio_dll.DioEchoBackByte
DioEchoBackByte.restype = ctypes.c_long
DioEchoBackByte.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioEchoBackBit(short id, short bit_no, unsigned char *data);
DioEchoBackBit = cdio_dll.DioEchoBackBit
DioEchoBackBit.restype = ctypes.c_long
DioEchoBackBit.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioSetDemoByte(short id, short port_no, unsigned char data);
DioSetDemoByte = cdio_dll.DioSetDemoByte
DioSetDemoByte.restype = ctypes.c_long
DioSetDemoByte.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_ubyte]

# C Prototype: long DioSetDemoBit(short id, short bit_no, unsigned char data);
DioSetDemoBit = cdio_dll.DioSetDemoBit
DioSetDemoBit.restype = ctypes.c_long
DioSetDemoBit.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_ubyte]

#----------------------------------------
# Multiple I/O function
#----------------------------------------
# C Prototype: long DioInpMultiByte(short id, short *port_no, short port_num, unsigned char *data);
DioInpMultiByte = cdio_dll.DioInpMultiByte
DioInpMultiByte.restype = ctypes.c_long
DioInpMultiByte.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioInpMultiBit(short id, short *bit_no, short bit_num, unsigned char *data);
DioInpMultiBit = cdio_dll.DioInpMultiBit
DioInpMultiBit.restype = ctypes.c_long
DioInpMultiBit.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioInpMultiByteSR(short id, short *port_no, short port_num, unsigned char *data, unsigned short *time_stamp, unsigned char mode);
DioInpMultiByteSR = cdio_dll.DioInpMultiByteSR
DioInpMultiByteSR.restype = ctypes.c_long
DioInpMultiByteSR.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ushort), ctypes.c_ubyte]

# C Prototype: long DioInpMultiBitSR(short id, short *bit_no, short bit_num, unsigned char *data, unsigned short *time_stamp, unsigned char mode);
DioInpMultiBitSR = cdio_dll.DioInpMultiBitSR
DioInpMultiBitSR.restype = ctypes.c_long
DioInpMultiBitSR.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ushort), ctypes.c_ubyte]

# C Prototype: long DioOutMultiByte(short id, short *port_no, short port_num, unsigned char *data);
DioOutMultiByte = cdio_dll.DioOutMultiByte
DioOutMultiByte.restype = ctypes.c_long
DioOutMultiByte.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioOutMultiBit(short id, short *bit_no, short bit_num, unsigned char *data);
DioOutMultiBit = cdio_dll.DioOutMultiBit
DioOutMultiBit.restype = ctypes.c_long
DioOutMultiBit.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioEchoBackMultiByte(short id, short *port_no, short port_num, unsigned char *data);
DioEchoBackMultiByte = cdio_dll.DioEchoBackMultiByte
DioEchoBackMultiByte.restype = ctypes.c_long
DioEchoBackMultiByte.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

# C Prototype: long DioEchoBackMultiBit(short id, short *bit_no, short bit_num, unsigned char *data);
DioEchoBackMultiBit = cdio_dll.DioEchoBackMultiBit
DioEchoBackMultiBit.restype = ctypes.c_long
DioEchoBackMultiBit.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

#----------------------------------------
# Interrupt function
#----------------------------------------
# C Prototype: long DioSetInterruptEvent(short id, short bit_no, short logic);
DioSetInterruptEvent = cdio_dll.DioSetInterruptEvent
DioSetInterruptEvent.restype = ctypes.c_long
DioSetInterruptEvent.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_short]

# C Prototype: long DioSetInterruptCallBackProc(short id, PDIO_INT_CALLBACK call_back, void *param);
DioSetInterruptCallBackProc = cdio_dll.DioSetInterruptCallBackProc
DioSetInterruptCallBackProc.restype = ctypes.c_long
DioSetInterruptCallBackProc.argtypes = [ctypes.c_short, PDIO_INT_CALLBACK, ctypes.c_void_p]

#----------------------------------------
# Trigger function
#----------------------------------------
# C Prototype: long DioSetTrgEvent(short id, short bit_no, short logic, long tim);
DioSetTrgEvent = cdio_dll.DioSetTrgEvent
DioSetTrgEvent.restype = ctypes.c_long
DioSetTrgEvent.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_long]

# C Prototype: long DioSetTrgCallBackProc(short id, PDIO_TRG_CALLBACK call_back, void *param);
DioSetTrgCallBackProc = cdio_dll.DioSetTrgCallBackProc
DioSetTrgCallBackProc.restype = ctypes.c_long
DioSetTrgCallBackProc.argtypes = [ctypes.c_short, PDIO_TRG_CALLBACK, ctypes.c_void_p]

#----------------------------------------
# Information function
#----------------------------------------
# C Prototype: long DioQueryDeviceName(short index, char *device_name, char *device);
DioQueryDeviceName = cdio_dll.DioQueryDeviceName
DioQueryDeviceName.restype = ctypes.c_long
DioQueryDeviceName.argtypes = [ctypes.c_short, ctypes.c_char_p, ctypes.c_char_p]

# C Prototype: long DioGetDeviceInfo(char *device, short info_type, void *param1, void *param2, void *param3);
DioGetDeviceInfo = cdio_dll.DioGetDeviceInfo
DioGetDeviceInfo.restype = ctypes.c_long
DioGetDeviceInfo.argtypes = [ctypes.c_char_p, ctypes.c_short, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

# C Prototype: long DioGetMaxPorts(short id, short *in_port_num, short *out_port_num);
DioGetMaxPorts = cdio_dll.DioGetMaxPorts
DioGetMaxPorts.restype = ctypes.c_long
DioGetMaxPorts.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short)]

#----------------------------------------
# Counter function
#----------------------------------------
# C Prototype: long DioGetMaxCountChannels(short Id, short *ChannelNum);
DioGetMaxCountChannels = cdio_dll.DioGetMaxCountChannels
DioGetMaxCountChannels.restype = ctypes.c_long
DioGetMaxCountChannels.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short)]

# C Prototype: long DioSetCountEdge(short Id, short *ChNo, short ChNum, short *CountEdge);
DioSetCountEdge = cdio_dll.DioSetCountEdge
DioSetCountEdge.restype = ctypes.c_long
DioSetCountEdge.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_short)]

# C Prototype: long DioGetCountEdge(short Id, short *ChNo, short ChNum, short *CountEdge);
DioGetCountEdge = cdio_dll.DioGetCountEdge
DioGetCountEdge.restype = ctypes.c_long
DioGetCountEdge.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_short)]

# C Prototype: long DioSetCountMatchValue(short Id, short *ChNo, short ChNum, short *CompareRegNo, unsigned int *CompareCount);
DioSetCountMatchValue = cdio_dll.DioSetCountMatchValue
DioSetCountMatchValue.restype = ctypes.c_long
DioSetCountMatchValue.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_uint)]

# C Prototype: long DioStartCount(short Id, short *ChNo, short ChNum);
DioStartCount = cdio_dll.DioStartCount
DioStartCount.restype = ctypes.c_long
DioStartCount.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short]

# C Prototype: long DioStopCount(short Id, short *ChNo, short ChNum);
DioStopCount = cdio_dll.DioStopCount
DioStopCount.restype = ctypes.c_long
DioStopCount.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short]

# C Prototype: long DioGetCountStatus(short Id, short *ChNo, short ChNum, unsigned int *CountStatus);
DioGetCountStatus = cdio_dll.DioGetCountStatus
DioGetCountStatus.restype = ctypes.c_long
DioGetCountStatus.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_uint)]

# C Prototype: long DioCountPreset(short Id, short *ChNo, short ChNum, unsigned int *PresetCount);
DioCountPreset = cdio_dll.DioCountPreset
DioCountPreset.restype = ctypes.c_long
DioCountPreset.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_uint)]

# C Prototype: long DioReadCount(short Id, short *ChNo, short ChNum, unsigned int *Count);
DioReadCount = cdio_dll.DioReadCount
DioReadCount.restype = ctypes.c_long
DioReadCount.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_uint)]

# C Prototype: long DioReadCountSR(short Id, short *ChNo, short ChNum, unsigned int *Count, unsigned short *Timestanp, unsigned char Mode);
DioReadCountSR = cdio_dll.DioReadCountSR
DioReadCountSR.restype = ctypes.c_long
DioReadCountSR.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short), ctypes.c_short, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_ushort), ctypes.c_ubyte]

#----------------------------------------
# DM function
#----------------------------------------
# C Prototype: long DioDmSetDirection(short id, unsigned long Direction)
DioDmSetDirection = cdio_dll.DioDmSetDirection
DioDmSetDirection.restype = ctypes.c_long
DioDmSetDirection.argtypes = [ctypes.c_short, ctypes.c_ulong]

# C Prototype: long DioDmSetStandAlone(short id);
DioDmSetStandAlone = cdio_dll.DioDmSetStandAlone
DioDmSetStandAlone.restype = ctypes.c_long
DioDmSetStandAlone.argtypes = [ctypes.c_short]

# C Prototype: long DioDmSetMasterCfg(short id, unsigned long ExtSig1, unsigned long ExtSig2, unsigned long ExtSig3,
#                                       unsigned long MasterHalt, unsigned long SlaveHalt);
DioDmSetMasterCfg = cdio_dll.DioDmSetMasterCfg
DioDmSetMasterCfg.restype = ctypes.c_long
DioDmSetMasterCfg.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetSlaveCfg(short id, unsigned long ExtSig1, unsigned long ExtSig2, unsigned long ExtSig3,
#                                       unsigned long MasterHalt, unsigned long SlaveHalt);
DioDmSetSlaveCfg = cdio_dll.DioDmSetSlaveCfg
DioDmSetSlaveCfg.restype = ctypes.c_long
DioDmSetSlaveCfg.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetStartTrg(short id, unsigned long Dir, unsigned long Start);
DioDmSetStartTrg = cdio_dll.DioDmSetStartTrg
DioDmSetStartTrg.restype = ctypes.c_long
DioDmSetStartTrg.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetStartPattern(short id, unsigned long Ptn, unsigned long Mask);
DioDmSetStartPattern = cdio_dll.DioDmSetStartPattern
DioDmSetStartPattern.restype = ctypes.c_long
DioDmSetStartPattern.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetClockTrg(short id, unsigned long Dir, unsigned long Clock);
DioDmSetClockTrg = cdio_dll.DioDmSetClockTrg
DioDmSetClockTrg.restype = ctypes.c_long
DioDmSetClockTrg.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetInternalClock(short id, unsigned long Dir, unsigned long Clock, unsigned long Unit);
DioDmSetInternalClock = cdio_dll.DioDmSetInternalClock
DioDmSetInternalClock.restype = ctypes.c_long
DioDmSetInternalClock.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetStopTrg(short id, unsigned long Dir, unsigned long Stop);
DioDmSetStopTrg = cdio_dll.DioDmSetStopTrg
DioDmSetStopTrg.restype = ctypes.c_long
DioDmSetStopTrg.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmSetStopNum(short id, unsigned long Dir, unsigned long StopNum);
DioDmSetStopNum = cdio_dll.DioDmSetStopNum
DioDmSetStopNum.restype = ctypes.c_long
DioDmSetStopNum.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmReset(short id, unsigned long Reset);
DioDmReset = cdio_dll.DioDmReset
DioDmReset.restype = ctypes.c_long
DioDmReset.argtypes = [ctypes.c_short, ctypes.c_ulong]

# C Prototype: long DioDmSetBuff(short id, unsigned long Dir, unsigned long *Buff, unsigned long Len, unsigned long IsRing);
DioDmSetBuff = cdio_dll.DioDmSetBuff
DioDmSetBuff.restype = ctypes.c_long
DioDmSetBuff.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_ulong]

# C Prototype: long DioDmStart(short id, unsigned long Dir);
DioDmStart = cdio_dll.DioDmStart
DioDmStart.restype = ctypes.c_long
DioDmStart.argtypes = [ctypes.c_short, ctypes.c_ulong]

# C Prototype: long DioDmStop(short id, unsigned long Dir);
DioDmStop = cdio_dll.DioDmStop
DioDmStop.restype = ctypes.c_long
DioDmStop.argtypes = [ctypes.c_short, ctypes.c_ulong]

# C Prototype: long DioDmGetStatus(short id, unsigned long Dir, unsigned long *Status, unsigned long *Err);
DioDmGetStatus = cdio_dll.DioDmGetStatus
DioDmGetStatus.restype = ctypes.c_long
DioDmGetStatus.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong)]

# C Prototype: long DioDmGetCount(short id, unsigned long Dir, unsigned long *Count, unsigned long *Carry);
DioDmGetCount = cdio_dll.DioDmGetCount
DioDmGetCount.restype = ctypes.c_long
DioDmGetCount.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong)]

# C Prototype: long DioDmGetWritePointerUserBuf(short id, unsigned long Dir, unsigned long *WritePointer, unsigned long *Count, unsigned long *Carry);
DioDmGetWritePointerUserBuf = cdio_dll.DioDmGetWritePointerUserBuf
DioDmGetWritePointerUserBuf.restype = ctypes.c_long
DioDmGetWritePointerUserBuf.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong)]

# C Prototype: long DioDmGetFifoCount(short id, unsigned long Dir, unsigned long *Count);
DioDmGetFifoCount = cdio_dll.DioDmGetFifoCount
DioDmGetFifoCount.restype = ctypes.c_long
DioDmGetFifoCount.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]

# C Prototype: long DioDmSetStopEvent(short id, unsigned long Dir, PDIO_STOP_CALLBACK CallBack, void *Parameter);
DioDmSetStopEvent = cdio_dll.DioDmSetStopEvent
DioDmSetStopEvent.restype = ctypes.c_long
DioDmSetStopEvent.argtypes = [ctypes.c_short, ctypes.c_ulong, PDIO_STOP_CALLBACK, ctypes.c_void_p]

# C Prototype: long DioDmSetCountEvent(short id, unsigned long Dir, unsigned long Count, PDIO_COUNT_CALLBACK CallBack, void *Parameter);
DioDmSetCountEvent = cdio_dll.DioDmSetCountEvent
DioDmSetCountEvent.restype = ctypes.c_long
DioDmSetCountEvent.argtypes = [ctypes.c_short, ctypes.c_ulong, ctypes.c_ulong, PDIO_COUNT_CALLBACK, ctypes.c_void_p]

#----------------------------------------
# Failsafe  function
#----------------------------------------
# C Prototype: long DioGetPatternEventStatus(short id, short *status);
DioGetPatternEventStatus = cdio_dll.DioGetPatternEventStatus
DioGetPatternEventStatus.restype = ctypes.c_long
DioGetPatternEventStatus.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_short)]

# C Prototype: long DioResetPatternEvent(short id, unsigned char *data);
DioResetPatternEvent = cdio_dll.DioResetPatternEvent
DioResetPatternEvent.restype = ctypes.c_long
DioResetPatternEvent.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_ubyte)]

