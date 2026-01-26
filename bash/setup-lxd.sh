#!/bin/bash
#\file    setup-lxd.sh
#\brief   Setup an LXD container
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.27, 2024

cnt_name_prefix="${1:-testenv}"

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

host_ver=$(lsb_release -rs)
lxc launch ubuntu:$host_ver $cnt_name
echo "Created $cnt_name as ubuntu:$host_ver"

# Disable autostart
lxc config set $cnt_name boot.autostart false

echo "Waiting for cloud-init to finish (network & user creation)..."
lxc exec $cnt_name -- cloud-init status --wait

lxc exec $cnt_name -- getent hosts google.com || echo "WARNING: DNS resolution might be failing."

lxc exec $cnt_name -- sudo apt update
lxc exec $cnt_name -- sudo apt install -y openssh-server

lxc exec $cnt_name -- pkill -KILL -u ubuntu || true

# Rename the default user
lxc exec $cnt_name -- usermod -l $user_name ubuntu
lxc exec $cnt_name -- groupmod -n $user_name ubuntu
lxc exec $cnt_name -- usermod -d /home/$user_name -m $user_name
lxc exec $cnt_name -- sed -i "s/ubuntu/$user_name/g" /etc/sudoers.d/90-cloud-init-users

echo "Set password for $user_name"
lxc exec $cnt_name -- passwd $user_name

echo "Configuring SSH..."
lxc exec $cnt_name -- rm -f /etc/ssh/sshd_config.d/50-cloud-init.conf
lxc exec $cnt_name -- rm -f /etc/ssh/sshd_config.d/60-cloudimg-settings.conf
lxc exec $cnt_name -- sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
lxc exec $cnt_name -- sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
lxc exec $cnt_name -- systemctl restart ssh


echo "########"
echo "In order to upload a file, you can do:"
echo "\$ lxc file push ~/prg/fvinstaller/fv+gripper/install.sh $cnt_name/home/$user_name/"
echo "########"


echo "Copying kernel modules and headers from host to container..."
echo "This usually takes a while due to the large number of files. Please wait."
if [[ -e /lib/modules/`uname -r` ]];then
  sudo lxc file push --recursive /lib/modules/`uname -r` $cnt_name/lib/modules/
fi
if [[ -e /usr/src/linux-headers-`uname -r` ]];then
  sudo lxc file push --recursive /usr/src/linux-headers-`uname -r` $cnt_name/usr/src/
fi
echo "Copy completed."

echo "########"
echo "In order to access the virtual env:"
echo "\$ lxc exec $cnt_name -t -- bash -c \"ssh $user_name@localhost\""
echo "########"
lxc exec $cnt_name -t -- bash -c "ssh $user_name@localhost"
