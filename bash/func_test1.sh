#!/bin/bash
#\file    func_test1.sh
#\brief   certain bash script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.26, 2024

# Dictionary to hold temporary user selections
declare -A user_selection_temporary

# It prompts the user to enter y or n
# and saves the user input to user_selection_temporary dictionary with the specified key.
# If the dictionary has already been set for the specified key, the prompt will be skipped.
# Note that unlike the usual ask, user_selection is not saved into a file.
# params:
#   $1 - key of user_selection dictionary
#   $2 - prompt
# return:
#   0: y, 1: n
ask_temporary() {
    local key_name="$1"
    shift 1

    get_temporary "$key_name"
    local retval=$?; if [[ $retval -ne 255 ]]; then return $retval; fi

    echo "$@"

    while true; do
        echo -n '  (y|n) > '
        read s
        if [[ "$s" == "y" ]] || [[ "$s" == "n" ]];then break; fi
    done

    # Save the input (only to the dictionary).
    user_selection_temporary["$key_name"]="$s"

    if [[ "$s" == "y" ]]; then return 0; fi
    if [[ "$s" == "n" ]]; then return 1; fi
}

# It gets the value specified by the key from user_selection_temporary dictionary.
# params:
#   $1 - key of user_selection_temporary dictionary
# return:
#   0: y, 1: n, -1: not exist
get_temporary() {
    s=${user_selection_temporary["$1"]}
    if [[ "$s" == "y" ]]; then return 0; fi
    if [[ "$s" == "n" ]]; then return 1; fi
    return -1
}


ask_temporary 'gripper_edit_config_sh' 'Do you want to run an editor?'

get_temporary gripper_edit_config_sh
if [[ $? -eq 0 ]]; then
  echo "Executing Nano"
fi


