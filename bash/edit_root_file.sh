#!/bin/bash
#\file    edit_root_file.sh
#\brief   Test let normal user to edit a specitic root file.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.24, 2021

# #NOTE:
# This script needs a root privelege, and the goal here is
# to let users execute this script without giving them the root privelege.
# A solution is edit /etc/sudoers by visudo:
# $ sudo visudo
# Then, add following lines.
# #To give a user akihikoy the permission to execute:
#   akihikoy ALL=PASSWD: ALL, NOPASSWD: /home/akihikoy/prg/ay_test/bash/edit_root_file.sh
# #To give a group dialout the permission to execute (% indicate a group):
#   %dialout ALL=PASSWD: ALL, NOPASSWD: /home/akihikoy/prg/ay_test/bash/edit_root_file.sh

echo "Test" | sudo tee /tmp/edit_root_file.txt


