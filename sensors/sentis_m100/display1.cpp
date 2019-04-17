/************************************************************************/
// Derived from Bluetechnix glut-distances sample program
/**
 * @file
 * @version     1.0.0
 *
 * \cond HIDDEN_SYMBOLS
 * 
 * Copyright (c) 2013
 * VoXel Interaction Design GmbH
 *
 * @author VoXel Interaction Design <office@voxel.at>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * \endcond
 *
 * @section DESCRIPTION
 *
 * Example application for the sentis-ToF-m100 camera API
 *
 * This application shows an OpenGl view which displays 3D images based in distance values.
 * It uses the distances + amplitudes data format.
 * The view displays a grid of 160cmX120cm and the X axis shows the depth ~300cm.
 *
 * The amplitude values are used to set the color (gray) intensity of the image.
 *
 */
/************************************************************************/

#ifdef __APPLE__
#include <GLUT/glut.h>
#define VSNPRINTF vsnprintf
#else
#if defined _WIN32 || defined _WIN64
#define _USE_MATH_DEFINES // VS losses some math macros as M_PI
#define VSNPRINTF vsnprintf_s
#include <windows.h>
#else
#define VSNPRINTF vsnprintf
#endif
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <stdlib.h>
#include <m100api.h>

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <iostream>
using namespace std;

#define IMAGE_WIDTH 160
#define IMAGE_HEIGHT 120
#define IMAGE_SIZE (IMAGE_WIDTH*IMAGE_HEIGHT)

// m100 registers
#define ModulationFrequency                     0x0009
#define LedboardTemp 				0x001B
#define CalibrationCommand                      0x000F
#define CalibrationExtended                     0x0021
#define FrameTime                               0x001F
#define TempCompGradient2                       0x0030
#define TempCompGradient3                       0x003C
#define BuildYearMonth                          0x003D
#define BuildDayHour                            0x003E
#define BuildMinuteSecond                       0x003F
#define UpTimeLow                               0x0040
#define UpTimeHigh                              0x0041
#define AkfPlausibilityCheckAmpLimit            0x0042
#define CommKeepAliveTimeout                    0x004E
#define CommKeepAliveReset                      0x004F
#define AecAvgWeight0                           0x01A9
#define AecAvgWeight1                           0x01AA
#define AecAvgWeight2                           0x01AB
#define AecAvgWeight3                           0x01AC
#define AecAvgWeight4                           0x01AD
#define AecAvgWeight5                           0x01AE
#define AecAvgWeight6                           0x01AF
#define AecAmpTarget                            0x01B0
#define AecTintStepMax                          0x01B1
#define AecTintMax                              0x01B2
#define AecKp                                   0x01B3
#define AecKi                                   0x01B4
#define AecKd                                   0x01B5

/************************************************************************/

unsigned char rgb_intensity_image[ 2*IMAGE_SIZE*3 ];
unsigned char rgb_depth_image[ 2*IMAGE_SIZE*3 ];
unsigned char rgb_double_wide_image[ 2*IMAGE_SIZE*3 ];
unsigned char rgb_double_wide_image_flipped[ 2*IMAGE_SIZE*3 ];
float fbuffer[2*IMAGE_SIZE];
float fbuffer2[2*IMAGE_SIZE];

//----- Camera control functions -----//
T_SENTIS_HANDLE handler;
T_ERROR_CODE error;
	// size is 76800bytes; IMAGE_SIZE*2images*2bytes (short)
static int size = 4*IMAGE_SIZE;
static unsigned short buffer[2*IMAGE_SIZE];

/************************************************************************/

/**
 *
 * @brief Helper method that starts the connection with the camera and set default 
 * image_data format.
 * 
 * @param [in,out] T_SENTIS_HANDLE * Camera handler. 
 *
 */
int initCamera(T_SENTIS_HANDLE *handle) {

	T_SENTIS_CONFIG config;	
	// Default network configuration
	// config.tcp_ip = "192.168.0.10";
	// config.tcp_ip = "192.168.111.21";
	config.tcp_ip = "192.168.0.14";
	config.udp_ip = "224.0.0.1";
	config.udp_port = 10002;
	config.tcp_port = 10001;
	config.flags = HOLD_CONTROL_ALIVE;

	// Connect with the device
	printf("CLIENT: open connection\n");
	*handle = STSopen(&config,&error);
	if (error != 0) {
		printf("Not possible to open the connection. Error: %d\n", error);
		exit(1);
	}
	
	unsigned short res;

	error = STSreadRegister(*handle, Mode0, &res, 0, 0);
	printf("Mode0: (1) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//Sleep(1);
	error = STSreadRegister(*handle, Status, &res, 0, 0);
	printf("Status: (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, ImageDataFormat, &res, 0, 0);
	printf("ImageDataFormat: (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, IntegrationTime, &res, 0, 0);
	printf("IntegrationTime: (min 50 max 25000) %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, DeviceType, &res, 0, 0);
	printf("DeviceType: (A9C1) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, DeviceInfo, &res, 0, 0);
	printf("DeviceInfo: (1001) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, FirmwareInfo, &res, 0, 0);
	printf("FirmwareInfo: (1001) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, ModulationFrequency, &res, 0, 0);
	printf("ModulationFrecuency: (7D0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, FrameRate, &res, 0, 0);
	printf("FrameRate %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, HardwareConfiguration, &res, 0, 0);
	printf("HardwareConfiguration (5A) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, SerialNumberLowWord, &res, 0, 0);
	printf("SerialNumberLowWord (1) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, SerialNumberHighWord, &res, 0, 0);
	printf("SerialNumberHighWord (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, FrameCounter, &res, 0, 0);
	printf("FrameCounter %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	error = STSreadRegister(*handle, CalibrationCommand, &res, 0, 0);
	printf("CalibrationCommand (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, ConfidenceThresLow, &res, 0, 0);
	printf("ConfidenceThresLow (12C) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, ConfidenceThresHig, &res, 0, 0);
	printf("ConfidenceThresHig (3A98) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, Mode1, &res, 0, 0);
	printf("Mode1 (800 or 808) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, CalculationTime, &res, 0, 0);
	printf("CalculationTime %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, LedboardTemp, &res, 0, 0);
	printf("LedboarsTemp %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, MainboardTemp, &res, 0, 0);
	printf("MainboardTemp %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	//sleep(1);
	error = STSreadRegister(*handle, LinearizationAmplitude, &res, 0, 0);
	printf("LinearizationAmplitude (190) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, LinearizationPhasseShift, &res, 0, 0);
	printf("LinearizationPhasseShift (1B58) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FrameTime, &res, 0, 0);
	printf("FrameTime %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, CalibrationExtended, &res, 0, 0);
	printf("CalibrationExtended (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, MaxLedTemp, &res, 0, 0);
	printf("MaxLedTemp %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, HorizontalFov, &res, 0, 0);
	printf("HorizontalFov %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, VerticalFov, &res, 0, 0);
	printf("VerticalFov %#X %d\n", res, res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, TriggerDelay, &res, 0, 0);
	printf("TriggerDelay (0) %#X\n", res );
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, BootloaderStatus, &res, 0, 0);
	printf("BootloaderStatus (4000) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, TemperatureCompensationGradient, &res, 0, 0);
	printf("TemperatureCompensationGradient (E12) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	//sleep(1);
	error = STSreadRegister(*handle, ApplicationVersion, &res, 0, 0);
	printf("ApplicationVersion (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}
	
	error = STSreadRegister(*handle, DistCalibGradient, &res, 0, 0);
	printf("DistCalibGradient (4000) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, TempCompGradient2, &res, 0, 0);
	printf("TempCompGradient2 (3A) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, CmdExec, &res, 0, 0);
	printf("CmdExec (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, CmdExecResult, &res, 0, 0);
	printf("CmdExecResult (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryMacAddr2, &res, 0, 0);
	printf("FactoryMacAddr2 (26) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryMacAddr1, &res, 0, 0);
	printf("FactoryMacAddr1 (3500) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryMacAddr0, &res, 0, 0);
	printf("FactoryMacAddr0 (4B) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryYear, &res, 0, 0);
	printf("FactoryYear (7DE) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryMonthDay, &res, 0, 0);
	printf("FactoryMonthDay (516) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryHourMinute, &res, 0, 0);
	printf("FactoryHourMinute (D34) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FactoryTimezone, &res, 0, 0);
	printf("FactoryTimezone (2) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, TempCompGradient3, &res, 0, 0);
	printf("TempCompGradient3 (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, BuildYearMonth, &res, 0, 0);
	printf("BuildYearMonth (7DE1) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, BuildDayHour, &res, 0, 0);
	printf("BuildDayHour (3E9) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, BuildMinuteSecond, &res, 0, 0);
	printf("BuildMinuteSecond (7CE) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, UpTimeLow, &res, 0, 0);
	printf("UpTimeLow %#X (0)\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, UpTimeHigh, &res, 0, 0);
	printf("UpTimeHigh (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AkfPlausibilityCheckAmpLimit, &res,
				0, 0);
	printf("AkfPlausibilityCheckAmpLimit (32) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, CommKeepAliveTimeout, &res, 0, 0);
	printf("CommKeepAliveTimeout (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, CommKeepAliveReset, &res, 0, 0);
	printf("CommKeepAliveReset (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight0, &res, 0, 0);
	printf("AecAvgWeight0 (4444) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight1, &res, 0, 0);
	printf("AecAvgWeight1 (44CC) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight2, &res, 0, 0);
	printf("AecAvgWeight2 (C44C) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight3, &res, 0, 0);
	printf("AecAvgWeight3 (FC44) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight4, &res, 0, 0);
	printf("AecAvgWeight4 (CCC4) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight5, &res, 0, 0);
	printf("AecAvgWeight5 (4444) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecAvgWeight6, &res, 0, 0);
	printf("AecAvgWeight6 (4000) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}


	error = STSreadRegister(*handle, AecAmpTarget, &res, 0, 0);
	printf("AecAmpTarget (2BC) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecTintStepMax, &res, 0, 0);
	printf("AecTintStepMax (21) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecTintMax, &res, 0, 0);
	printf("AecTintMax (2710) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecKp, &res, 0, 0);
	printf("AecKp (28) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecKi, &res, 0, 0);
	printf("AecKi (F) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, AecKd, &res, 0, 0);
	printf("AecKd (0) %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, Mode1, &res, 0, 0);
	printf("Mode1 %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	// set bit 3
	printf( "Setting auto exposure.\n" );
	error = STSwriteRegister(*handle, Mode1, (res | 0x8) );
	if (error != 0) {
		printf("Error when writing register. Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, Mode1, &res, 0, 0);
	printf("Mode1 %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FrameRate, &res, 0, 0);
	printf("FrameRate %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	printf( "Setting frame rate.\n" );
	error = STSwriteRegister(*handle, FrameRate, 40 ); // or 40 or whatever
	if (error != 0) {
		printf("Error when writing register. Error: %d\n", error);
		exit(1);
	}

	error = STSreadRegister(*handle, FrameRate, &res, 0, 0);
	printf("FrameRate %#X\n", res);
	if (error != 0) {
		printf("Error: %d\n", error);
		exit(1);
	}

	// Set image_data format to distances+amplitudes
	error = STSwriteRegister(*handle, ImageDataFormat, DEPTH_AMP_DATA );
	if (error != 0) {
		printf("Error when writing register. Error: %d\n", error);
		exit(1);
	}

	return 0;
}

/************************************************************************/

/**
 *
 * @brief Get a frame from the camera and save it in the global buffer variable.
 * 
 * @param [in] T_SENTIS_HANDLE Camera handler. 
 *
 */
int getFrame(T_SENTIS_HANDLE handle) {
	
	T_SENTIS_DATA_HEADER header;
	// size is 76800bytes; IMAGE_SIZE*2images*2bytes (short)
	size = (int)sizeof(buffer);
	error = STSgetData(handle, &header,  (char *)buffer, &size, 0, 0);
	if (error != 0) {
		printf("Error getting frame. Error: %d\n", error);
		return -1;
	}
	return 0;
}

//----- END Camera control functions END-----//

/************************************************************************/

/**
 *
 * @brief Called every time a window is resized to resize the projection matrix
 *
 */
void my_reshape( int w, int h )
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D( 0.0, (GLfloat) IMAGE_WIDTH, 0.0, (GLfloat) IMAGE_HEIGHT );
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glViewport(0,0,w,h);
}


/************************************************************************/

#define N_JET 6

float jet_red[N_JET][2] = { { 0.000, 0.0 },
			    { 0.125, 0.0 },
			    { 0.375, 0.0 },
			    { 0.625, 1.0 },
			    { 0.875, 1.0 },
			    { 1.000, 0.5 } };

float jet_green[N_JET][2] = { { 0.000, 0.0 },
			      { 0.125, 0.0 },
			      { 0.375, 1.0 },
			      { 0.625, 1.0 },
			      { 0.875, 0.0 },
			      { 1.000, 0.0 } };

float jet_blue[N_JET][2] = { { 0.000, 0.5 },
			     { 0.125, 1.0 },
			     { 0.375, 1.0 },
			     { 0.625, 0.0 },
			     { 0.875, 0.0 },
			     { 1.000, 0.0 } };

void jet( float fvalue, unsigned char *value )
{
  int i, i0, i1;
  float w0, w1;
  
  if ( fvalue <= 0.0 )
    {
      value[0] = jet_red[0][1];
      value[1] = jet_green[0][1];
      value[2] = jet_blue[0][1];
      return;
    }

  if ( fvalue >= 1.0 )
    {
      value[0] = jet_red[N_JET-1][1];
      value[1] = jet_green[N_JET-1][1];
      value[2] = jet_blue[N_JET-1][1];
      return;
    }

  // assume last value is taken
  i0 = N_JET - 2;
  i1 = N_JET - 1;
  w0 = 0.0;
  w1 = 1.0;
  for ( i = 1; i < N_JET; i++ )
    {
      if ( fvalue < jet_red[i][0] )
	{
	  i0 = i - 1;
	  i1 = i;
	  w0 = (jet_red[i][0] - fvalue)/(jet_red[i][0] - jet_red[i-1][0]);
	  w1 = 1.0 - w0;
	  break;
	}
    }

  value[0] = 255*(w0*jet_red[i0][1] + w1*jet_red[i1][1]);
  value[1] = 255*(w0*jet_green[i0][1] + w1*jet_green[i1][1]);
  value[2] = 255*(w0*jet_blue[i0][1] + w1*jet_blue[i1][1]);

  /*
  value[0] = 255*fvalue;
  value[1] = 0;
  value[2] = 0;
  */
}

/************************************************************************/

void process_depth_image()
{
  int i;
  unsigned short *p_image;
  float *p_fimage;
  float fvalue, fvalue2;
  unsigned char value[3];
  /* f1, f2
  float cutoff1 = 350.0;
  float cutoff2 = 850.0;
  */
  /*
  float cutoff1 = 200.0;
  float cutoff2 = 700.0;
  */
  float cutoff1 = 20.0;
  float cutoff2 = 700.0;

  p_image = buffer;

  for ( i = 0; i < IMAGE_SIZE*3; i += 3 )
    {
      fvalue = (float) (*p_image++);
      if ( fvalue <= cutoff1 )
	{ // white
	  value[0] = 255;
	  value[1] = 255;
	  value[2] = 255;
	}
      else if ( fvalue >= cutoff2 )
	{ // black
	  value[0] = 0;
	  value[1] = 0;
	  value[2] = 0;
	}
      else
	{
	  fvalue2 = cutoff2 - fvalue;
	  if ( fvalue2 <= 0 )
	    { // black
	      value[0] = 0;
	      value[1] = 0;
	      value[2] = 0;
	    }
	  else if ( fvalue2 >= (cutoff2 - cutoff1) )
	    { // white
	      value[0] = 255;
	      value[1] = 255;
	      value[2] = 255;
	    }
	  else
	    {
	      fvalue2 /= (cutoff2 - cutoff1);
	      jet( fvalue2, value );
	    }
	}
      rgb_depth_image[ i ] = value[0];
      rgb_depth_image[ i+1 ] = value[1];
      rgb_depth_image[ i+2 ] = value[2];
    }
}

/************************************************************************/

void process_intensity_image()
{
  int i = 0;
  int ix, iy;
  unsigned short *p_image;
  float *p_fimage;
  float *p_fimage2;
  double fvalue, fvalue2, fvalue3;
  unsigned char pixel[3];
  double cutoff1 = 0.0;
  double cutoff2 = 21028.0;
  double min_fvalue = 1e6;
  double max_fvalue = -1e6;
  double min_fvalue2 = 1e6;
  double max_fvalue2 = -1e6;

  p_image = buffer + IMAGE_SIZE;
  p_fimage = fbuffer + IMAGE_SIZE;

  // Pass 0: try log transform
  for ( iy = 0; iy < IMAGE_HEIGHT; iy++ )
    {
      for ( ix = 0; ix < IMAGE_WIDTH; ix++ )
	{
	  fvalue = (float) (*p_image++);
	  if ( fvalue <= cutoff1 )
	    fvalue = cutoff1;
	  else if ( fvalue >= cutoff2 )
	    fvalue = cutoff2;
	  *p_fimage++ = log( fvalue );
	}
    }

  p_fimage = fbuffer + IMAGE_SIZE;

  // Pass 1: find bounds
  for ( iy = 0; iy < IMAGE_HEIGHT; iy++ )
    {
      for ( ix = 0; ix < IMAGE_WIDTH; ix++ )
	{
	  fvalue = *p_fimage++;
	  if ( fvalue <= cutoff1 )
	    fvalue = cutoff1;
	  else if ( fvalue >= cutoff2 )
	    fvalue = cutoff2;

	  if ( fvalue < min_fvalue )
	    min_fvalue = fvalue;
	  if ( fvalue > max_fvalue )
	    max_fvalue = fvalue;
	}
    }
  printf( "i1: min: %g; max: %g\n", min_fvalue, max_fvalue );

  p_fimage = fbuffer + IMAGE_SIZE;
  p_fimage2 = fbuffer2 + IMAGE_SIZE;
  cutoff1 = min_fvalue;
  cutoff2 = max_fvalue;

  // Pass 2: apply bounds
  for ( iy = 0; iy < IMAGE_HEIGHT; iy++ )
    {
      for ( ix = 0; ix < IMAGE_WIDTH; ix++ )
	{
	  fvalue = *p_fimage++;
	  if ( fvalue <= cutoff1 )
	    { 
	      // black
	      pixel[0] = 0;
	      pixel[1] = 0;
	      pixel[2] = 0;
	    }
	  else if ( fvalue >= cutoff2 )
	    { 
	      // white
	      pixel[0] = 255;
	      pixel[1] = 255;
	      pixel[2] = 255;
	    }
	  else
	    {
	      fvalue2 = 255*(fvalue - cutoff1)/(cutoff2 - cutoff1);
	      if ( fvalue2 < min_fvalue2 )
		min_fvalue2 = fvalue2;
	      if ( fvalue2 > max_fvalue2 )
		max_fvalue2 = fvalue2;
	      fvalue3 = fvalue2;
	      if ( fvalue3 < 0.0 )
		fvalue3 = 0;
	      if ( fvalue3 > 255.0 )
		fvalue3 = 255.0;
	      pixel[0] = (unsigned char) fvalue3;
	      pixel[1] = pixel[0];
	      pixel[2] = pixel[0];
	    }
	  rgb_intensity_image[ i ] = pixel[0];
	  rgb_intensity_image[ i+1 ] = pixel[1];
	  rgb_intensity_image[ i+2 ] = pixel[2];
	  i += 3;
	}
    }
  printf( "i2: min: %g; max: %g\n", min_fvalue2, max_fvalue2 );
}

/************************************************************************/

void create_double_wide()
{
  int ix, iy;
  unsigned char *p_rgb_left, *p_rgb_right, *p_rgb_dw, *p_rgb_dwf;

  p_rgb_left = rgb_intensity_image;
  p_rgb_right = rgb_depth_image;
  p_rgb_dw = rgb_double_wide_image;

  for ( iy = 0; iy < IMAGE_HEIGHT; iy++ )
    {
      for ( ix = 0; ix < 3*IMAGE_WIDTH; ix++ )
	*p_rgb_dw++ = *p_rgb_left++;
      for ( ix = 0; ix < 3*IMAGE_WIDTH; ix++ )
	*p_rgb_dw++ = *p_rgb_right++;
    }

  p_rgb_dw = rgb_double_wide_image;

  for ( iy = 0; iy < IMAGE_HEIGHT; iy++ )
    {
      p_rgb_dwf = rgb_double_wide_image_flipped
	+ (IMAGE_HEIGHT - iy)*2*IMAGE_WIDTH*3;
      for ( ix = 0; ix < 6*IMAGE_WIDTH; ix++ )
	*p_rgb_dwf++ = *p_rgb_dw++;
    }
}

/************************************************************************/

/**
 *
 * @brief Called at the start of the program, after a glutPostRedisplay() and during idle
 * to display a frame
 *
 */
// need argument for timer version
// void my_display(int value) {
void my_display() {
		
	if(getFrame(handler) == -1)
	  {
 	    cout << "*Frame fail." << endl;
            exit( 1 );
	  }
	
	process_depth_image();
	process_intensity_image();
	create_double_wide();

	glClear( GL_COLOR_BUFFER_BIT );
	glDrawPixels( 2*IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGB,
		      GL_UNSIGNED_BYTE, rgb_double_wide_image_flipped );
    	glFlush();

	glutSwapBuffers();
	
	// glutTimerFunc( 1000, display, 0 );

	return;
}

/************************************************************************/

int main(int argc, char **argv)
{
  int i;
  unsigned char value;

  for ( i = 0; i < 2*IMAGE_SIZE*3; i += 3 )
    {
      value = (i/3) % 256;
      rgb_double_wide_image[i] = value;
      rgb_double_wide_image[i+1] = value;
      rgb_double_wide_image[i+2] = value;
    }

    glutInit(&argc, argv); // Initializes glut
    
    // Sets up a double buffer with RGB components
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    
    glutInitWindowSize( 2*IMAGE_WIDTH, IMAGE_HEIGHT );
    
    // Sets the window position to the upper left
    glutInitWindowPosition(0, 0);
    
    // Creates a window using internal glut functionality
    glutCreateWindow( "Intensity and Depth" );
    
    // passes reshape and display functions to the OpenGL machine for callback
    glutReshapeFunc( my_reshape );
    glutDisplayFunc( my_display );
    glutIdleFunc( my_display );

    if(initCamera(&handler) == -1)
		exit(1);

    // Starts the program.
    glutMainLoop();

    return 0;
}

