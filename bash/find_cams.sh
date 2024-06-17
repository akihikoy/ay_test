#!/bin/bash
#\file    find_cams.sh
#\brief   Find camera devices with a device type names.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.17, 2024

# Specify the target device name
TARGET_DEVICE_NAME="VGA Camera"

# List all devices and their paths
DEVICES=$(v4l2-ctl --list-devices 2>&1)
DEVICE_STATUS=$?

# Initialize an empty array to hold matching device paths
MATCHING_PATHS=()

# Read through the devices output
while IFS= read -r line; do
# echo "debug,${line}"
  if [[ "$line" == *"$TARGET_DEVICE_NAME"* ]]; then
    # Get the next line for the path
    read -r path_line
    # Extract the path
    DEVICE_PATH=$(echo $path_line | grep -oP '(?<=/dev/).*')

    # Check if the device path exists in /dev/v4l/by-path/
    for by_path in /dev/v4l/by-path/*; do
      if [ "$(readlink -f $by_path)" == "/dev/$DEVICE_PATH" ]; then
        MATCHING_PATHS+=("$by_path")
      fi
    done
  fi
done <<< "$DEVICES"

# List the matched paths
for path in "${MATCHING_PATHS[@]}"; do
  echo "$path"
done

