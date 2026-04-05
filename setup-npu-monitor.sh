#!/bin/bash
#
# setup-npu-monitor.sh
#
# Install the Intel NPU System Monitoring Tool.
# Run this on the Proxmox host (not inside the LXC container),
# since it needs direct access to /sys/class/intel_pmt/.
#
# Usage:
#   chmod +x setup-npu-monitor.sh
#   ./setup-npu-monitor.sh
#

set -euo pipefail

INSTALL_DIR="/opt/npu-monitor"
REPO_URL="https://github.com/open-edge-platform/edge-ai-libraries.git"

echo "==> Installing Intel NPU Monitor Tool to ${INSTALL_DIR}..."

if [[ -d "${INSTALL_DIR}" ]]; then
    echo "==> Already installed, updating..."
    cd "${INSTALL_DIR}"
    git pull
else
    git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
    git sparse-checkout set tools/npu-monitor-tool
fi

chmod +x "${INSTALL_DIR}/tools/npu-monitor-tool/npu-monitor-tool.py"

# Create a convenience symlink
ln -sf "${INSTALL_DIR}/tools/npu-monitor-tool/npu-monitor-tool.py" /usr/local/bin/npu-monitor

echo ""
echo "==> NPU Monitor installed."
echo ""
echo "    Usage (interactive on host):"
echo "      sudo npu-monitor              # one-shot snapshot"
echo "      sudo npu-monitor -i 1000      # continuous (1s interval)"
echo "      sudo npu-monitor --csv -i 1000  # export to CSV"
echo ""
echo "    To expose metrics to LXC containers via HTTP:"
echo "      ./setup-npu-monitor-service.sh"
