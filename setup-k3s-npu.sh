#!/bin/bash
#
# setup-k3s-npu.sh
#
# Run this inside a fresh privileged LXC container on Proxmox to set up:
#   1. System prerequisites for K3s in LXC
#   2. K3s (lightweight Kubernetes)
#   3. Helm
#   4. nerdctl (container build tool for containerd/K3s)
#   5. Node Feature Discovery (NFD)
#   6. cert-manager
#   7. Intel Device Plugin Operator + NPU Plugin
#
# Prerequisites (done on the Proxmox HOST before running this script):
#   - LXC created with: nesting=1, fuse=1, privileged
#   - LXC config includes: apparmor unconfined, cgroup device allow, mount auto
#   - /dev/accel/accel0 bind-mounted into the container
#   - udev rule on host: chmod 666 /dev/accel/accel0
#   - See SETUP.md sections 2.2-2.5 for full host-side setup
#
# Usage:
#   chmod +x setup-k3s-npu.sh
#   ./setup-k3s-npu.sh
#

set -euo pipefail

# --- Configuration ---
K3S_DISABLE_TRAEFIK=true
K3S_DISABLE_SERVICELB=true
NFD_VERSION="0.16.4"
CERTMANAGER_VERSION="v1.15.2"
PANTHER_LAKE_DEVICE_ID="b03e"
HELM_TIMEOUT="120s"
K3S_READY_TIMEOUT=120      # seconds to wait for K3s node to be Ready
POD_READY_TIMEOUT=180      # seconds to wait for pods to be Running
NPU_REGISTER_TIMEOUT=60    # seconds to wait for NPU to appear in allocatable

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helpers ---
log()   { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        fail "This script must be run as root"
    fi
}

check_lxc() {
    if [[ ! -f /proc/1/environ ]] || ! grep -q container=lxc /proc/1/environ 2>/dev/null; then
        # Alternative check
        if [[ ! -d /proc/1/ns ]] || systemd-detect-virt -c -q 2>/dev/null; then
            warn "Cannot confirm this is an LXC container. Proceeding anyway..."
        fi
    fi
}

check_npu_device() {
    if [[ ! -e /dev/accel/accel0 ]]; then
        fail "/dev/accel/accel0 not found. Ensure NPU passthrough is configured in the LXC config on the Proxmox host. See SETUP.md section 2.3."
    fi

    if ! python3 -c "import os; os.open('/dev/accel/accel0', os.O_RDWR)" 2>/dev/null; then
        # Try a simpler check
        if ! cat /dev/accel/accel0 >/dev/null 2>&1; then
            warn "/dev/accel/accel0 exists but may not be accessible. Check permissions on the Proxmox host (chmod 666 /dev/accel/accel0)."
        fi
    fi
    ok "/dev/accel/accel0 is present"
}

wait_for_k3s_ready() {
    local timeout=$1
    local elapsed=0
    log "Waiting up to ${timeout}s for K3s node to be Ready..."
    while [[ $elapsed -lt $timeout ]]; do
        if kubectl get nodes 2>/dev/null | grep -q " Ready "; then
            ok "K3s node is Ready"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    fail "K3s node did not become Ready within ${timeout}s. Check: systemctl status k3s"
}

wait_for_pods_ready() {
    local namespace=$1
    local label=$2
    local timeout=$3
    local elapsed=0
    log "Waiting up to ${timeout}s for pods (${label}) in ${namespace}..."
    while [[ $elapsed -lt $timeout ]]; do
        local total running
        total=$(kubectl get pods -n "$namespace" -l "$label" --no-headers 2>/dev/null | wc -l)
        running=$(kubectl get pods -n "$namespace" -l "$label" --no-headers 2>/dev/null | grep -cE "Running|Completed" || true)
        if [[ $total -gt 0 && $total -eq $running ]]; then
            ok "All pods ($label) in $namespace are running"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    warn "Some pods ($label) in $namespace are not ready after ${timeout}s"
    kubectl get pods -n "$namespace" -l "$label" 2>/dev/null || true
    return 1
}

wait_for_all_pods_namespace() {
    local namespace=$1
    local timeout=$2
    local elapsed=0
    log "Waiting up to ${timeout}s for all pods in ${namespace}..."
    while [[ $elapsed -lt $timeout ]]; do
        local not_ready
        not_ready=$(kubectl get pods -n "$namespace" --no-headers 2>/dev/null | grep -cvE "Running|Completed" || true)
        if [[ $not_ready -eq 0 ]]; then
            local total
            total=$(kubectl get pods -n "$namespace" --no-headers 2>/dev/null | wc -l)
            if [[ $total -gt 0 ]]; then
                ok "All $total pods in $namespace are running"
                return 0
            fi
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    warn "Some pods in $namespace are not ready after ${timeout}s"
    kubectl get pods -n "$namespace" --no-headers 2>/dev/null || true
    return 1
}

# ============================================================================
# STEP 0: Pre-flight checks
# ============================================================================
echo ""
echo "============================================================"
echo "  Intel NUC16 Panther Lake: K3s + NPU Setup Script"
echo "============================================================"
echo ""

check_root
check_lxc

# ============================================================================
# STEP 1: System prerequisites
# ============================================================================
log "STEP 1: Installing system prerequisites..."

# Update packages
apt-get update -qq || fail "apt-get update failed"
apt-get upgrade -y -qq || warn "apt-get upgrade had warnings"

# Install required packages
apt-get install -y -qq \
    curl \
    wget \
    gnupg2 \
    ca-certificates \
    apt-transport-https \
    open-iscsi \
    nfs-common \
    jq \
    make \
    || fail "Failed to install prerequisites"

ok "System packages installed"

# K3s requires /dev/kmsg
if [[ ! -e /dev/kmsg ]]; then
    ln -s /dev/console /dev/kmsg
    log "Created /dev/kmsg -> /dev/console symlink"
fi

# Persist across reboots
cat > /etc/rc.local << 'RCLOCAL'
#!/bin/sh -e
if [ ! -e /dev/kmsg ]; then
    ln -s /dev/console /dev/kmsg
fi
mount --make-rshared /
exit 0
RCLOCAL
chmod +x /etc/rc.local
systemctl enable rc-local 2>/dev/null || true
systemctl start rc-local 2>/dev/null || true

mount --make-rshared / 2>/dev/null || warn "mount --make-rshared failed (may already be set)"
swapoff -a 2>/dev/null || true

if ! echo "$PATH" | grep -q "/usr/local/bin"; then
    export PATH="$PATH:/usr/local/bin"
fi

# bashrc setup
grep -q 'usr/local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
grep -q 'KUBECONFIG' ~/.bashrc 2>/dev/null || echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc
grep -q 'alias k=kubectl' ~/.bashrc 2>/dev/null || {
    echo 'source <(kubectl completion bash)' >> ~/.bashrc
    echo 'alias k=kubectl' >> ~/.bashrc
    echo 'complete -o default -F __start_kubectl k' >> ~/.bashrc
}

ok "System prerequisites configured"

check_npu_device

# ============================================================================
# STEP 2: Install K3s
# ============================================================================
log "STEP 2: Installing K3s..."

if command -v k3s &>/dev/null && systemctl is-active --quiet k3s 2>/dev/null; then
    ok "K3s is already installed and running, skipping installation"
else
    K3S_FLAGS="--write-kubeconfig-mode 644 --snapshotter=native"
    K3S_FLAGS="$K3S_FLAGS --kubelet-arg=feature-gates=KubeletInUserNamespace=true"
    K3S_FLAGS="$K3S_FLAGS --kube-controller-manager-arg=feature-gates=KubeletInUserNamespace=true"
    K3S_FLAGS="$K3S_FLAGS --kube-apiserver-arg=feature-gates=KubeletInUserNamespace=true"

    if [[ "$K3S_DISABLE_TRAEFIK" == true ]]; then
        K3S_FLAGS="$K3S_FLAGS --disable=traefik"
    fi
    if [[ "$K3S_DISABLE_SERVICELB" == true ]]; then
        K3S_FLAGS="$K3S_FLAGS --disable=servicelb"
    fi

    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server $K3S_FLAGS" sh - \
        || fail "K3s installation failed"

    ok "K3s installed"
fi

# Set up kubeconfig for this session
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# Wait for K3s to be ready
wait_for_k3s_ready $K3S_READY_TIMEOUT

# Show node info
kubectl get nodes
echo ""

# ============================================================================
# STEP 3: Install Helm
# ============================================================================
log "STEP 3: Installing Helm..."

if command -v helm &>/dev/null; then
    ok "Helm is already installed ($(helm version --short 2>/dev/null))"
else
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash \
        || fail "Helm installation failed"
    ok "Helm installed"
fi

# ============================================================================
# STEP 4: Install nerdctl (container build tool for containerd/K3s)
# ============================================================================
log "STEP 4: Installing nerdctl..."

if command -v nerdctl &>/dev/null; then
    ok "nerdctl is already installed ($(nerdctl --version 2>/dev/null))"
else
    NERDCTL_VERSION=$(curl -sL https://api.github.com/repos/containerd/nerdctl/releases/latest | jq -r .tag_name | sed 's/^v//')
    if [[ -z "$NERDCTL_VERSION" || "$NERDCTL_VERSION" == "null" ]]; then
        fail "Could not determine latest nerdctl version from GitHub API"
    fi

    NERDCTL_URL="https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-full-${NERDCTL_VERSION}-linux-amd64.tar.gz"
    log "Downloading nerdctl v${NERDCTL_VERSION}..."

    wget -q "$NERDCTL_URL" -O /tmp/nerdctl.tar.gz \
        || fail "Failed to download nerdctl from $NERDCTL_URL"

    FILESIZE=$(stat -c%s /tmp/nerdctl.tar.gz 2>/dev/null || echo "0")
    if [[ "$FILESIZE" -lt 1000 ]]; then
        rm -f /tmp/nerdctl.tar.gz
        fail "Downloaded nerdctl archive is too small (${FILESIZE} bytes) -- likely a bad URL or redirect"
    fi

    tar -xzf /tmp/nerdctl.tar.gz -C /usr/local/ \
        || fail "Failed to extract nerdctl"
    rm -f /tmp/nerdctl.tar.gz

    if ! command -v nerdctl &>/dev/null; then
        fail "nerdctl not found in PATH after installation"
    fi

    ok "nerdctl v${NERDCTL_VERSION} installed"
fi

# ============================================================================
# STEP 5: Add Helm repos
# ============================================================================
log "STEP 5: Adding Helm repositories..."

helm repo add nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts 2>/dev/null || true
helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
helm repo add intel https://intel.github.io/helm-charts/ 2>/dev/null || true
helm repo update || fail "Helm repo update failed"

ok "Helm repositories configured"

# ============================================================================
# STEP 6: Install Node Feature Discovery (NFD)
# ============================================================================
log "STEP 6: Installing Node Feature Discovery..."

if helm status nfd -n node-feature-discovery &>/dev/null; then
    ok "NFD is already installed, skipping"
else
    helm install nfd nfd/node-feature-discovery \
        --namespace node-feature-discovery \
        --create-namespace \
        --version "$NFD_VERSION" \
        --wait \
        --timeout "$HELM_TIMEOUT" \
        || fail "NFD installation failed"
    ok "NFD installed"
fi

wait_for_all_pods_namespace "node-feature-discovery" "$POD_READY_TIMEOUT"

# Wait for NFD to label the node (takes a few seconds after pods start)
log "Waiting for NFD to detect hardware..."
sleep 10

# Verify NFD detected the NPU via PCI
if kubectl get node -o json | jq -e '.items[].metadata.labels["feature.node.kubernetes.io/pci-1200_8086.present"]' &>/dev/null; then
    ok "NFD detected Intel NPU (PCI class 1200, vendor 8086)"
else
    warn "NFD has not labeled the node with PCI-1200_8086. The NPU may not be detected."
    warn "Check NFD worker logs: kubectl logs -n node-feature-discovery -l app.kubernetes.io/component=worker"
fi

# ============================================================================
# STEP 7: Install cert-manager
# ============================================================================
log "STEP 7: Installing cert-manager..."

if helm status cert-manager -n cert-manager &>/dev/null; then
    ok "cert-manager is already installed, skipping"
else
    helm install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --version "$CERTMANAGER_VERSION" \
        --set installCRDs=true \
        --wait \
        --timeout "$HELM_TIMEOUT" \
        || fail "cert-manager installation failed"
    ok "cert-manager installed"
fi

wait_for_all_pods_namespace "cert-manager" "$POD_READY_TIMEOUT"

# ============================================================================
# STEP 8: Install Intel Device Plugin Operator
# ============================================================================
log "STEP 8: Installing Intel Device Plugin Operator..."

if helm status dp-operator -n inteldeviceplugins-system &>/dev/null; then
    ok "Intel Device Plugin Operator is already installed, skipping"
else
    helm install dp-operator intel/intel-device-plugins-operator \
        --namespace inteldeviceplugins-system \
        --create-namespace \
        --wait \
        --timeout "$HELM_TIMEOUT" \
        || fail "Intel Device Plugin Operator installation failed"
    ok "Intel Device Plugin Operator installed"
fi

wait_for_all_pods_namespace "inteldeviceplugins-system" "$POD_READY_TIMEOUT"

# ============================================================================
# STEP 9: Install Intel NPU Device Plugin
# ============================================================================
log "STEP 9: Installing Intel NPU Device Plugin..."

if helm status npu -n inteldeviceplugins-system &>/dev/null; then
    ok "Intel NPU Device Plugin is already installed, skipping"
else
    helm install npu intel/intel-device-plugins-npu \
        --namespace inteldeviceplugins-system \
        --create-namespace \
        --set nodeFeatureRule=true \
        || fail "Intel NPU Device Plugin installation failed"
    ok "Intel NPU Device Plugin installed"
fi

# ============================================================================
# STEP 10: Patch NodeFeatureRule for Panther Lake
# ============================================================================
log "STEP 10: Checking NodeFeatureRule for Panther Lake device ID ($PANTHER_LAKE_DEVICE_ID)..."

# Wait for the NodeFeatureRule to exist
ELAPSED=0
while [[ $ELAPSED -lt 30 ]]; do
    if kubectl get nodefeaturerule intel-dp-npu-device &>/dev/null; then
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if ! kubectl get nodefeaturerule intel-dp-npu-device &>/dev/null; then
    fail "NodeFeatureRule 'intel-dp-npu-device' not found. The NPU Helm chart may not have created it."
fi

# Check if the Panther Lake device ID is already in the rule
EXISTING_IDS=$(kubectl get nodefeaturerule intel-dp-npu-device -o json \
    | jq -r '.spec.rules[0].matchFeatures[0].matchExpressions.device.value[]' 2>/dev/null)

if echo "$EXISTING_IDS" | grep -q "$PANTHER_LAKE_DEVICE_ID"; then
    ok "Panther Lake device ID ($PANTHER_LAKE_DEVICE_ID) already in NodeFeatureRule"
else
    log "Adding Panther Lake device ID ($PANTHER_LAKE_DEVICE_ID) to NodeFeatureRule..."
    kubectl patch nodefeaturerule intel-dp-npu-device --type='json' -p="[
        {\"op\": \"add\", \"path\": \"/spec/rules/0/matchFeatures/0/matchExpressions/device/value/-\", \"value\": \"$PANTHER_LAKE_DEVICE_ID\"}
    ]" || fail "Failed to patch NodeFeatureRule"
    ok "Panther Lake device ID added to NodeFeatureRule"
fi

# Wait for NFD to apply the label
log "Waiting for NFD to apply intel.feature.node.kubernetes.io/npu=true label..."
ELAPSED=0
while [[ $ELAPSED -lt 60 ]]; do
    if kubectl get node -o json | jq -e '.items[].metadata.labels["intel.feature.node.kubernetes.io/npu"]' &>/dev/null; then
        ok "NPU label applied to node"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if ! kubectl get node -o json | jq -e '.items[].metadata.labels["intel.feature.node.kubernetes.io/npu"]' &>/dev/null; then
    warn "NPU label not applied after 60s. The NPU DaemonSet may not schedule."
    warn "Debug: kubectl get nodefeaturerule intel-dp-npu-device -o yaml"
fi

# ============================================================================
# STEP 11: Verify NPU is schedulable
# ============================================================================
log "STEP 11: Waiting for NPU to register as a schedulable resource..."

# Wait for the NPU plugin pod to start
sleep 10
wait_for_all_pods_namespace "inteldeviceplugins-system" "$POD_READY_TIMEOUT"

# Wait for the NPU resource to appear
ELAPSED=0
while [[ $ELAPSED -lt $NPU_REGISTER_TIMEOUT ]]; do
    NPU_COUNT=$(kubectl get node -o json | jq -r '.items[0].status.allocatable["npu.intel.com/accel"] // empty' 2>/dev/null)
    if [[ -n "$NPU_COUNT" && "$NPU_COUNT" != "0" ]]; then
        ok "NPU is schedulable: npu.intel.com/accel = $NPU_COUNT"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [[ -z "${NPU_COUNT:-}" || "${NPU_COUNT:-}" == "0" ]]; then
    warn "NPU resource not found in node allocatable after ${NPU_REGISTER_TIMEOUT}s"
    warn "Debug steps:"
    warn "  kubectl get pods -n inteldeviceplugins-system"
    warn "  kubectl logs -n inteldeviceplugins-system -l app=intel-npu-plugin"
    warn "  kubectl get node -o json | jq '.items[].status.allocatable'"
fi

# ============================================================================
# STEP 12: Run NPU test pod
# ============================================================================
log "STEP 12: Running NPU device test pod..."

# Clean up any previous test pod
kubectl delete pod npu-test --ignore-not-found=true 2>/dev/null
sleep 2

cat <<'TESTPOD' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: npu-test
spec:
  restartPolicy: Never
  containers:
  - name: npu-test
    image: ubuntu:24.04
    command: ["/bin/bash", "-c"]
    args:
    - |
      echo "=== NPU Device Test ==="
      echo "Checking for /dev/accel devices..."
      ls -la /dev/accel/ 2>/dev/null || echo "ERROR: /dev/accel not found"
      echo ""
      echo "Device info:"
      cat /sys/class/accel/accel0/device/uevent 2>/dev/null || echo "Cannot read device info"
      echo ""
      echo "NPU test complete."
    resources:
      limits:
        npu.intel.com/accel: 1
      requests:
        npu.intel.com/accel: 1
TESTPOD

# Wait for the test pod to complete
log "Waiting for test pod to complete..."
ELAPSED=0
while [[ $ELAPSED -lt 120 ]]; do
    PHASE=$(kubectl get pod npu-test -o jsonpath='{.status.phase}' 2>/dev/null)
    if [[ "$PHASE" == "Succeeded" ]]; then
        break
    elif [[ "$PHASE" == "Failed" ]]; then
        warn "Test pod failed"
        break
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

echo ""
echo "--- Test pod output ---"
kubectl logs npu-test 2>/dev/null || warn "Could not get test pod logs"
echo "--- End test pod output ---"
echo ""

# Check if test passed
if kubectl logs npu-test 2>/dev/null | grep -q "/dev/accel/accel0"; then
    ok "NPU device is accessible from Kubernetes pods!"
else
    warn "NPU device was not visible in the test pod"
fi

# Clean up test pod
kubectl delete pod npu-test --ignore-not-found=true 2>/dev/null

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  Components installed:"
echo "    - K3s (Kubernetes)"
echo "    - Helm"
echo "    - Node Feature Discovery (NFD)"
echo "    - cert-manager"
echo "    - Intel Device Plugin Operator"
echo "    - Intel NPU Device Plugin"
echo ""
echo "  NPU resource: npu.intel.com/accel"
echo ""
echo "  Quick commands:"
echo "    kubectl get nodes"
echo "    kubectl get pods -A"
echo "    kubectl get node -o json | jq '.items[].status.allocatable'"
echo ""
echo "  To run an OpenVINO workload with NPU access, add to your pod spec:"
echo "    resources:"
echo "      limits:"
echo "        npu.intel.com/accel: 1"
echo ""
echo "  NOTE: Source your shell config to pick up aliases:"
echo "    source ~/.bashrc"
echo ""
