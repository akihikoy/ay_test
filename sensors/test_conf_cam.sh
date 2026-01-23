#!/bin/bash
#\file    test_conf_cam.sh
#\brief   Camera exposure configuration test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.23, 2026

DEV="/dev/video4"

#Manual exposure with very large exposure.
uvcdynctrl -d $DEV -s "Exposure, Auto" 1
uvcdynctrl -d $DEV -s "Exposure (Absolute)" 1000
sleep 0.5

#Auto exposure ON (Aperture Priority Mode)
uvcdynctrl -d $DEV -s "Exposure, Auto" 3

uvcdynctrl -d $DEV -s "Brightness" 127
uvcdynctrl -d $DEV -s "Gain" 36
sleep 1.0

#Auto exposure OFF to fix the internal parameters
uvcdynctrl -d $DEV -s "Exposure, Auto" 1
sleep 0.5

uvcdynctrl -d $DEV -s "Gain" 0
uvcdynctrl -d $DEV -s "Brightness" 0
uvcdynctrl -d $DEV -s "Exposure (Absolute)" 100
sleep 0.2

