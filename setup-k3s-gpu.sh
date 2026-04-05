#!/bin/bash
#
# setup-k3s-gpu.sh
#
# Install the Intel GPU device plugin for K3s.
# Run this AFTER setup-k3s-npu.sh (which installs the Device Plugin Operator).
#
# This script:
#   1. Verifies the Device Plugin Operator is installed
#   2. Verifies /dev/dri is available
#   3. Installs the Intel GPU device plugin via Helm
#   4. Waits for the GPU to appear as an allocatable resource
#
# After this, pods can request gpu.intel.com/xe (Panther Lake)
# or gpu.intel.com/i915 (older platforms).
#
# Usage:
#   chmod +x setup-k3s-gpu.sh
#   ./setup-k3s-gpu.sh
#

set -euo pipefail

# Ensure KUBECONFIG and PATH are set (may not be in non-login shells)
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
export PATH=$PATH:/usr/local/bin

# --- Configuration ---
HELM_TIMEOUT="120s"
POD_READY_TIMEOUT=120
GPU_REGISTER_TIMEOUT=60
SHARED_DEV_NUM=1  # how many pods can share the GPU

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Helpers ---
log()   { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ============================================================================
# Preflight checks
# ============================================================================

if [[ $EUID -ne 0 ]]; then
    fail "This script must be run as root"
fi

if ! helm status dp-operator -n inteldeviceplugins-system &>/dev/null; then
    fail "Intel Device Plugin Operator not found. Run setup-k3s-npu.sh first."
fi
ok "Device Plugin Operator is installed"

if [[ ! -d /dev/dri ]]; then
    fail "/dev/dri not found. Ensure GPU passthrough is configured in the LXC config."
fi

if [[ ! -e /dev/dri/renderD128 ]]; then
    warn "/dev/dri/renderD128 not found. GPU may not be properly passed through."
fi
ok "/dev/dri is available"

# ============================================================================
# Install Intel GPU Device Plugin
# ============================================================================
log "Installing Intel GPU Device Plugin..."

if helm status intel-gpu-plugin -n inteldeviceplugins-system &>/dev/null; then
    ok "Intel GPU Device Plugin is already installed, skipping"
else
    helm install intel-gpu-plugin intel/intel-device-plugins-gpu \
        --namespace inteldeviceplugins-system \
        --set sharedDevNum=$SHARED_DEV_NUM \
        --set nodeFeatureRule=true \
        --wait \
        --timeout "$HELM_TIMEOUT" \
        || fail "Intel GPU Device Plugin installation failed"
    ok "Intel GPU Device Plugin installed"
fi

# ============================================================================
# Patch NodeFeatureRule to fix kernel.enabledmodule error
# ============================================================================
# The intel-device-plugins-gpu Helm chart (v0.35.0) installs a NodeFeatureRule
# that checks for both kernel.loadedmodule and kernel.enabledmodule to detect
# the i915/xe GPU driver. However, kernel.enabledmodule is not available in
# NFD 0.16.x, which causes the entire rule to fail with:
#   "failed to process rule: feature kernel.enabledmodule not available"
# This prevents NFD from setting the intel.feature.node.kubernetes.io/gpu=true
# label, which means the GPU device plugin daemonset never schedules (its node
# selector requires that label).
#
# Fix: patch the rule to only use kernel.loadedmodule (which works), then
# restart NFD master so it re-evaluates the patched rule.
log "Patching GPU NodeFeatureRule..."

ELAPSED=0
while [[ $ELAPSED -lt 30 ]]; do
    if kubectl get nodefeaturerule intel-dp-gpu-device &>/dev/null; then
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

kubectl patch nodefeaturerule intel-dp-gpu-device --type=json -p '[
  {"op": "replace", "path": "/spec/rules/0/matchAny", "value": [
    {"matchFeatures": [{"feature": "kernel.loadedmodule", "matchExpressions": {"i915": {"op": "Exists"}}}]},
    {"matchFeatures": [{"feature": "kernel.loadedmodule", "matchExpressions": {"xe": {"op": "Exists"}}}]}
  ]}
]' || warn "Failed to patch NodeFeatureRule"

kubectl rollout restart deployment nfd-node-feature-discovery-master -n node-feature-discovery
ok "NodeFeatureRule patched and NFD master restarted"

# Wait for NFD to apply the label
log "Waiting for NFD to apply GPU label..."
ELAPSED=0
while [[ $ELAPSED -lt 60 ]]; do
    GPU_LABEL=$(kubectl get node -o json | jq -r '.items[0].metadata.labels["intel.feature.node.kubernetes.io/gpu"] // empty')
    if [[ -n "$GPU_LABEL" ]]; then
        ok "GPU label applied by NFD"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [[ -z "$GPU_LABEL" ]]; then
    warn "NFD did not apply GPU label. Applying manually..."
    kubectl label node "$(hostname)" intel.feature.node.kubernetes.io/gpu=true
    ok "GPU label applied manually"
fi

# ============================================================================
# Wait for GPU to appear as allocatable resource
# ============================================================================
log "Waiting for GPU to appear in node allocatable resources..."

ELAPSED=0
while [[ $ELAPSED -lt $GPU_REGISTER_TIMEOUT ]]; do
    GPU_COUNT=$(kubectl get node -o json | jq -r '
        .items[0].status.allocatable["gpu.intel.com/xe"] //
        .items[0].status.allocatable["gpu.intel.com/i915"] //
        "0"')
    if [[ "$GPU_COUNT" != "0" && "$GPU_COUNT" != "null" ]]; then
        ok "GPU registered with $GPU_COUNT device(s) available"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    log "  Waiting... ($ELAPSED/$GPU_REGISTER_TIMEOUT seconds)"
done

if [[ $ELAPSED -ge $GPU_REGISTER_TIMEOUT ]]; then
    warn "GPU did not appear in allocatable resources within $GPU_REGISTER_TIMEOUT seconds"
    warn "Check: kubectl get node -o json | jq '.items[0].status.allocatable'"
    warn "The plugin may need more time, or the GPU may not be properly exposed."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
log "GPU Device Plugin setup complete."
echo ""

# Show what got registered
XE=$(kubectl get node -o json | jq -r '.items[0].status.allocatable["gpu.intel.com/xe"] // empty')
I915=$(kubectl get node -o json | jq -r '.items[0].status.allocatable["gpu.intel.com/i915"] // empty')

if [[ -n "$XE" ]]; then
    ok "Resource: gpu.intel.com/xe = $XE"
    echo ""
    echo "  To request GPU in a pod:"
    echo "    resources:"
    echo "      limits:"
    echo "        gpu.intel.com/xe: 1"
fi

if [[ -n "$I915" ]]; then
    ok "Resource: gpu.intel.com/i915 = $I915"
    echo ""
    echo "  To request GPU in a pod:"
    echo "    resources:"
    echo "      limits:"
    echo "        gpu.intel.com/i915: 1"
fi

if [[ -z "$XE" && -z "$I915" ]]; then
    warn "No GPU resources detected yet. Check again in a minute:"
    echo "    kubectl get node -o json | jq '.items[0].status.allocatable' | grep gpu"
fi
