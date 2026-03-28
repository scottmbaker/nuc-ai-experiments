#!/bin/bash
#
# setup-ssh.sh
#
# Run this inside a fresh LXC container to set up SSH access with a non-root user.
#
# Usage:
#   chmod +x setup-ssh.sh
#   ./setup-ssh.sh
#

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root"
    exit 1
fi

echo "Installing openssh-server and sudo and other stuff..."
apt-get update -qq
apt-get install -y -qq openssh-server sudo emacs-nox net-tools make  

echo "Enabling SSH..."
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
systemctl enable --now ssh

echo "Creating user smbaker..."
if id smbaker &>/dev/null; then
    echo "User smbaker already exists, skipping creation"
else
    useradd -m -s /bin/bash -u 1026 smbaker
    echo "smbaker:smbaker" | chpasswd
    usermod -aG sudo smbaker
    echo "smbaker ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/smbaker
    chmod 440 /etc/sudoers.d/smbaker
fi

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "Done! SSH into this container with:"
echo "  ssh smbaker@${IP}"
echo ""
echo "Remember to change the default password after first login:"
echo "  passwd"
