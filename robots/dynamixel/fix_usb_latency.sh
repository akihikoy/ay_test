#!/bin/bash
#\file    fix_usb_latency.sh
#\brief   Fix the latency issue of USB.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.01, 2018

#Usage: ./fix_usb_latency.sh ttyUSB0

dev=ttyUSB0
if [ $# -ge 1 ];then
  dev=$1
fi
if [ -f /sys/bus/usb-serial/devices/$dev/latency_timer ] && [ `cat /sys/bus/usb-serial/devices/$dev/latency_timer` -gt 1 ];then
  echo "Fixing the latency issue of $dev..."
  echo 1 | sudo tee /sys/bus/usb-serial/devices/$dev/latency_timer
fi
