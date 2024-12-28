#!/bin/bash
#\file    setup-lxd.sh
#\brief   Setup an LXD container
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.27, 2024

for i in $(seq 1 100); do
  cnt_name="fvinstalltest$i"
  if lxc list --format csv -c n | grep -wq $cnt_name; then
    echo "Container $cnt_name exists."
  else
    break
  fi
done

lxc launch ubuntu:18.04 $cnt_name

lxc exec $cnt_name -- sudo adduser fv
lxc exec $cnt_name -- sudo usermod -g fv -G fv,adm,cdrom,sudo,dip,video,plugdev,users fv
lxc exec $cnt_name -- sudo apt update
lxc exec $cnt_name -- sudo apt install -y openssh-server

echo ''
echo 'Configure as follows. Okay?
  PasswordAuthentication yes
  PubkeyAuthentication yes'
read
lxc exec $cnt_name -- sudo nano /etc/ssh/sshd_config
lxc exec $cnt_name -- systemctl enable ssh
lxc exec $cnt_name -- systemctl start ssh
lxc exec $cnt_name -- systemctl restart ssh

lxc file push ~/prg/fvinstaller/fv+gripper/install.sh $cnt_name/home/fv/

lxc exec $cnt_name -- bash -c 'ssh fv@localhost'

