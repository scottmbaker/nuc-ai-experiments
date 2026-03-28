# NUC16 Experiments

This repository contains experiments performed on an ASUS NUC16. The configuration is:

* NUC16 Pro with 356H CPU
* 64GB DDR5
* 1TB Samsung 9100 NVME drive

The notes in this README are somewhat terse, intended to help me remember what I did.

## Proxmox installation

Installed Proxmox 9.1 from USB stick. Use the non-graphical installer.

## Post-install setup proxmox

```bash
mv /etc/apt/sources.list.d/pve-enterprise.sources /etc/apt/sources.list.d/pve-enterprise.sources.disabled
mv /etc/apt/sources.list.d/ceph.sources /etc/apt/sources.list.d/ceph.sources.disabled

cat > /etc/apt/sources.list.d/pve-no-subscription.sources << 'EOF'
Types: deb
URIs: http://download.proxmox.com/debian/pve
Suites: trixie
Components: pve-no-subscription
EOF

sudo apt install emacs-nox
emacs /etc/default/grub
# add to KERNEL_CMDLINE: intel_iommu=on iommu=pt
reboot

chmod 666 /dev/accel/accel0
echo 'SUBSYSTEM=="accel", KERNEL=="accel*", MODE="0666"' > /etc/udev/rules.d/99-intel-npu.rules
udevadm control --reload-rules
udevadm trigger /dev/accel/accel0
```

## Create the LXC container for NPU experiments

Download the Ubuntu 24.04 LXC image.

```bash
pveam update
pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst
```

Create the LXC container and set the configuration to allow NPU passthrough.

```bash
cat > /root/setup-ssh.sh
   # paste the setup-ssh.sh script here

cat > /root/setup-k3s-npu.sh
   # paste the script here

pct create 100 local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst \
  --hostname k3s-node \
  --memory 16384 \
  --swap 0 \
  --cores 8 \
  --rootfs local-lvm:400 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp,hwaddr=BC:24:11:00:01:00 \
  --features nesting=1,fuse=1 \
  --ostype ubuntu \
  --password smbaker

cat >> /etc/pve/lxc/100.conf << 'EOF'
# K3s requirements
lxc.apparmor.profile: unconfined
lxc.cgroup.devices.allow: a
lxc.cgroup2.devices.allow: a
lxc.cap.drop:
lxc.mount.auto: proc:rw sys:rw cgroup:rw
# NPU passthrough
lxc.cgroup2.devices.allow: c 261:* rwm
lxc.mount.entry: /dev/accel dev/accel none bind,optional,create=dir
lxc.seccomp.profile:
EOF

pct start 100
pct push 100 /root/setup-k3s-npu.sh /root/setup-k3s-npu.sh
pct push 100 /root/setup-ssh.sh /root/setup-ssh.sh
```

## Enter the LXC container and setup SSH, K3s, and other things

```bash
pct enter 100

chmod +x /root/setup-ssh.sh
./setup-ssh.sh

# do the NPU install stuff
chmod +x /root/setup-k3s-npu.sh
./setup-k3s-npu.sh
```

At this point the LXC will have SSH running inside, making it
convient to SSH in as well as to SCP in files.

## Copy the files in npu-chatbot into the container

First, copy all the files in the npu-chatbot directory. Could
use SSH, or could copy paste them, or could copy from the proxmox
host. Any approach that gets them there.

Next, build, deploy, and attach:

```bash
# list models
make models

# build a chatbot for a particular model
sudo make build MODEL=qwen3-4b

# deploy the chatbot we built
make deploy MODEL=qwen3-4b

# attach to the chatbot to chat
make attach
```

You can also build all models, though it will take some time

```bash
# Build all models
make build-all
```
