#!/bin/bash -x
echo 'Execute "hcitool scan" to find ROBOTIS BT-210'
sudo rfcomm connect -i B8:63:BC:00:2E:84
