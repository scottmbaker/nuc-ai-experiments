MODEL ?= tinyllama
DEVICE ?= NPU
IMAGE_BASE := localhost/npu-chatbot
POD_NAME := npu-chatbot
CONTAINERD_SOCK := /run/k3s/containerd/containerd.sock
NERDCTL := nerdctl --address $(CONTAINERD_SOCK) --namespace k8s.io

# Parse model config
MODEL_LINE := $(shell grep "^$(MODEL)|" models.conf)
MODEL_ID := $(word 2,$(subst |, ,$(MODEL_LINE)))
MODEL_DIR := $(word 3,$(subst |, ,$(MODEL_LINE)))
IMAGE_NAME := $(IMAGE_BASE):$(MODEL)

.PHONY: buildkit build build-all deploy undeploy nuke attach logs status fix-k3s clean models help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

models: ## List available models
	@echo "Available models (use MODEL=<name>):"
	@echo ""
	@printf "  %-16s %s\n" "NAME" "HUGGING FACE MODEL"
	@printf "  %-16s %s\n" "----" "------------------"
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir; do \
		printf "  %-16s %s\n" "$$name" "$$hf"; \
	done
	@echo ""
	@echo "Note: Coder models (qwen25-coder-*) may have issues on NPU at this time."
	@echo "      Use DEVICE=CPU or DEVICE=GPU: make deploy MODEL=qwen25-coder-7b DEVICE=CPU"
	@echo ""
	@echo "Example: make build MODEL=qwen3-1.7b"

buildkit: ## Start buildkitd if not already running
	@if pgrep -x buildkitd >/dev/null; then \
		echo "buildkitd is already running"; \
	else \
		echo "Starting buildkitd..."; \
		buildkitd &>/dev/null & \
		sleep 2; \
		if pgrep -x buildkitd >/dev/null; then \
			echo "buildkitd started"; \
		else \
			echo "ERROR: buildkitd failed to start"; exit 1; \
		fi; \
	fi

build: buildkit ## Build image for a model (MODEL=<name>)
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "ERROR: Unknown model '$(MODEL)'. Run 'make models' to see available models."; \
		exit 1; \
	fi
	@echo "Building $(IMAGE_NAME)..."
	@echo "  Model: $(MODEL_ID)"
	@echo "  Dir:   $(MODEL_DIR)"
	$(NERDCTL) build \
		--build-arg MODEL_ID=$(MODEL_ID) \
		--build-arg MODEL_DIR=$(MODEL_DIR) \
		-t $(IMAGE_NAME) .

build-all: buildkit ## Build images for all models
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir; do \
		echo ""; \
		echo "========== Building $$name =========="; \
		echo "  Model: $$hf"; \
		$(NERDCTL) build \
			--build-arg MODEL_ID=$$hf \
			--build-arg MODEL_DIR=$$dir \
			-t $(IMAGE_BASE):$$name . || \
		echo "WARN: $$name failed to build, continuing..."; \
	done
	@echo ""
	@echo "All builds complete."

deploy: ## Deploy a model (MODEL=<name>)
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "ERROR: Unknown model '$(MODEL)'. Run 'make models' to see available models."; \
		exit 1; \
	fi
	@kubectl delete pod $(POD_NAME) --wait=true --ignore-not-found=true 2>/dev/null
	@echo "Cleaning up orphaned containerd state..."
	@for cid in $$(k3s crictl ps -a -q 2>/dev/null); do \
		if k3s crictl inspect $$cid 2>/dev/null | grep -q $(POD_NAME); then \
			echo "  Removing orphaned container $$cid"; \
			k3s crictl rm -f $$cid 2>/dev/null; \
		fi; \
	done
	@for pid in $$(k3s crictl pods -q 2>/dev/null); do \
		if k3s crictl inspectp $$pid 2>/dev/null | grep -q $(POD_NAME); then \
			echo "  Removing orphaned sandbox $$pid"; \
			k3s crictl stopp $$pid 2>/dev/null; \
			k3s crictl rmp $$pid 2>/dev/null; \
		fi; \
	done
	@sed 's|IMAGE_PLACEHOLDER|$(IMAGE_NAME)|; s|MODEL_DIR_PLACEHOLDER|$(MODEL_DIR)|; s|DEVICE_PLACEHOLDER|$(DEVICE)|' \
		npu-chatbot-pod.yaml | kubectl apply -f -
	@echo ""
	@echo "Waiting for pod to start..."
	@kubectl wait --for=condition=Ready pod/$(POD_NAME) --timeout=600s 2>/dev/null || \
		(echo "Pod not ready. Check: make logs"; exit 1)
	@echo ""
	@echo "============================================================"
	@echo "  Chatbot is ready! ($(MODEL))"
	@echo ""
	@echo "    make attach"
	@echo ""
	@echo "  Type 'quit' to exit, 'reset' for new conversation."
	@echo "============================================================"

undeploy: ## Remove the chatbot pod
	kubectl delete pod $(POD_NAME) --ignore-not-found=true

nuke: ## Kill orphaned containerd state for this pod
	@echo "Removing orphaned containers..."
	@for cid in $$(sudo k3s crictl ps -a -q 2>/dev/null); do \
		if sudo k3s crictl inspect $$cid 2>/dev/null | grep -q $(POD_NAME); then \
			echo "  Removing container $$cid"; \
			sudo k3s crictl rm -f $$cid 2>/dev/null; \
		fi; \
	done
	@echo "Removing orphaned sandboxes..."
	@for pid in $$(sudo k3s crictl pods -q 2>/dev/null); do \
		if sudo k3s crictl inspectp $$pid 2>/dev/null | grep -q $(POD_NAME); then \
			echo "  Removing sandbox $$pid"; \
			sudo k3s crictl stopp $$pid 2>/dev/null; \
			sudo k3s crictl rmp $$pid 2>/dev/null; \
		fi; \
	done
	@echo "Done."

attach: ## Attach to the running chatbot
	kubectl attach -it $(POD_NAME)

logs: ## Show chatbot pod logs
	kubectl logs $(POD_NAME) -f

status: ## Show pod status and NPU allocatable resources
	@kubectl get pod $(POD_NAME) 2>/dev/null || echo "Pod not running"
	@echo ""
	@echo "NPU resources:"
	@kubectl get node -o json | jq -r '.items[0].status.allocatable["npu.intel.com/accel"] // "not available"'
	@echo ""
	@echo "Built images:"
	@$(NERDCTL) images 2>/dev/null | grep npu-chatbot || echo "  (none)"

fix-k3s: ## Increase k3s timeouts and lower eviction thresholds (run once)
	@sudo mkdir -p /etc/rancher/k3s
	@echo 'kubelet-arg:' | sudo tee /etc/rancher/k3s/config.yaml
	@echo '  - "eviction-hard=imagefs.available<5%,nodefs.available<5%"' | sudo tee -a /etc/rancher/k3s/config.yaml
	@echo '  - "runtime-request-timeout=30m"' | sudo tee -a /etc/rancher/k3s/config.yaml
	@echo "Restarting k3s..."
	@sudo systemctl restart k3s
	@echo "Done. Wait ~30s for k3s to be ready."

clean: ## Remove all npu-chatbot images
	@$(NERDCTL) images 2>/dev/null | grep npu-chatbot | awk '{print $$1":"$$2}' | \
		xargs -r $(NERDCTL) rmi 2>/dev/null || true
	@echo "Cleaned up images"
