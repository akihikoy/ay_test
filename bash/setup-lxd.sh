#!/bin/bash
#\file    setup-lxd.sh
#\brief   Setup an LXD container
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.27, 2024

cnt_name_prefix="${1:-testenv}"
# user_name="${2:-fv}"

read -p "Enter username to setup [default: fv]: " input_user_name
user_name="${input_user_name:-fv}"

for i in $(seq 1 100); do
  cnt_name="${cnt_name_prefix}${i}"
  if lxc list --format csv -c n | grep -wq $cnt_name; then
    echo "Container $cnt_name exists."
  else
    break
  fi
done

#lxc launch ubuntu:18.04 $cnt_name
host_ver=$(lsb_release -rs)
lxc launch ubuntu:$host_ver $cnt_name
echo "Created $cnt_name as ubuntu:$host_ver"

#Disable autostart
lxc config set $cnt_name boot.autostart false

lxc exec $cnt_name -- sudo apt update
lxc exec $cnt_name -- sudo apt install -y openssh-server

# lxc exec $cnt_name -- sudo adduser fv
# lxc exec $cnt_name -- sudo usermod -g fv -G fv,adm,cdrom,sudo,dip,video,plugdev,users fv
# target_groups=$(lxc exec $cnt_name -- id -Gn ubuntu | tr ' ' ',')
# lxc exec $cnt_name -- sudo usermod -g fv -G "$target_groups" fv

#Rename the default user
lxc exec $cnt_name -- usermod -l $user_name ubuntu
lxc exec $cnt_name -- groupmod -n $user_name ubuntu
lxc exec $cnt_name -- usermod -d /home/$user_name -m $user_name
lxc exec $cnt_name -- sed -i "s/ubuntu/$user_name/g" /etc/sudoers.d/90-cloud-init-users
echo "Set password for $user_name"
lxc exec $cnt_name -- passwd $user_name

echo ''
echo 'Configure as follows. Okay?
  PasswordAuthentication yes
  PubkeyAuthentication yes'
read
lxc exec $cnt_name -- sudo nano /etc/ssh/sshd_config
lxc exec $cnt_name -- systemctl enable ssh
lxc exec $cnt_name -- systemctl start ssh
lxc exec $cnt_name -- systemctl restart ssh

echo "In order to upload a file, you can do:"
echo "lxc file push ~/prg/fvinstaller/fv+gripper/install.sh $cnt_name/home/$user_name/"

if [[ -e /lib/modules/`uname -r` ]];then
  sudo lxc file push --recursive /lib/modules/`uname -r` $cnt_name/lib/modules/
fi
if [[ -e /usr/src/linux-headers-`uname -r` ]];then
  sudo lxc file push --recursive /usr/src/linux-headers-`uname -r` $cnt_name/usr/src/
fi

lxc exec $cnt_name -- bash -c 'ssh $user_name@localhost'

