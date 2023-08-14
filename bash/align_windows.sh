#!/bin/bash
#\file    align_windows.sh
#\brief   Test of aligning windows with wmctrl.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.14, 2023

if ! command -v wmctrl &> /dev/null; then
  echo "wmctrl is not installed. Please install it first."
  exit 1
fi

get_window_id_by_title()
{
  wmctrl -l | grep -i "$1" | awk '{print $1}'
}

declare -A positions
positions["fvp_1_l-blob"]="0,0,640,480"
positions["fvp_1_r-blob"]="648,0,640,480"
positions["fvp_1_l-pxv"]="0,545,640,480"
positions["fvp_1_r-pxv"]="648,545,640,480"
positions["Robot Operation Panel"]="1200,0,600,500"

for app in "${!positions[@]}"; do
  window_id=$(get_window_id_by_title "$app")
  if [ -z "$window_id" ]; then
    echo "No window with title containing \"$app\" found."
    continue
  fi
  wmctrl -i -r "$window_id" -e "0,${positions[$app]}"
done

echo "Windows have been aligned."

