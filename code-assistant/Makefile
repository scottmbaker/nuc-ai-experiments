MODEL ?= qwen3-coder-30b

IMAGE_BASE       := localhost/code-assistant
POD_NAME         := code-assistant
CONTAINERD_SOCK  := /run/k3s/containerd/containerd.sock
NERDCTL          := sudo nerdctl --address $(CONTAINERD_SOCK) --namespace k8s.io
HELM_RELEASE     := code-assistant

# Parse model config (NAME|HF_MODEL_ID|MODEL_DIR)
MODEL_LINE := $(shell grep "^$(MODEL)|" models.conf)
MODEL_ID   := $(word 2,$(subst |, ,$(MODEL_LINE)))
MODEL_DIR  := $(word 3,$(subst |, ,$(MODEL_LINE)))
IMAGE_NAME := $(IMAGE_BASE):$(MODEL)

.PHONY: help models buildkit build deploy undeploy nuke logs status bench smoke clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

models: ## List available LLM models
	@echo "Available models (use MODEL=<name>):"
	@echo ""
	@printf "  %-22s %s\n" "NAME" "HUGGING FACE MODEL"
	@printf "  %-22s %s\n" "----" "------------------"
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir; do \
		printf "  %-22s %s\n" "$$name" "$$hf"; \
	done
	@echo ""
	@echo "Example: make build MODEL=qwen3-coder-30b"

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

build: buildkit ## Build OVMS image with model baked in (MODEL=qwen3-coder-30b)
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "ERROR: Unknown model '$(MODEL)'. Run 'make models' to see available models."; \
		exit 1; \
	fi
	@echo "Building $(IMAGE_NAME)..."
	@echo "  Source: $(MODEL_ID)"
	@echo "  Output: $(MODEL_DIR)"
	$(NERDCTL) build \
		--build-arg MODEL_ID=$(MODEL_ID) \
		--build-arg MODEL_DIR=$(MODEL_DIR) \
		-t $(IMAGE_NAME) .

bench: ## Run feasibility benchmark on the iGPU (requires exported model dir at MODEL_PATH)
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Usage: make bench MODEL_PATH=/path/to/exported/model [DEVICE=GPU]"; \
		exit 1; \
	fi
	python3 bench/bench.py --model "$(MODEL_PATH)" --device "$${DEVICE:-GPU}"

deploy: ## Deploy code-assistant pod (MODEL=qwen3-coder-30b)
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "ERROR: Unknown model '$(MODEL)'. Run 'make models' to see available models."; \
		exit 1; \
	fi
	@echo "Deploying $(MODEL)"
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
	helm upgrade --install $(HELM_RELEASE) ./chart \
		--set image=$(IMAGE_NAME)
	@echo ""
	@echo "Waiting for pod to start..."
	@kubectl wait --for=condition=Ready pod/$(POD_NAME) --timeout=600s 2>/dev/null || \
		(echo "Pod not ready. Check: make logs"; exit 1)
	@echo ""
	@NODE_IP=$$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'); \
	echo "============================================================"; \
	echo "  code-assistant ready! ($(MODEL))"; \
	echo ""; \
	echo "  REST (OpenAI-compatible):  http://$$NODE_IP:30083"; \
	echo "  gRPC:                       $$NODE_IP:30093"; \
	echo ""; \
	echo "  First request triggers GPU compile — typically ~15 seconds."; \
	echo "  Tail with 'make logs' to monitor progress."; \
	echo ""; \
	echo "  Smoke test:  make smoke ENDPOINT=http://$$NODE_IP:30083"; \
	echo "============================================================"

undeploy: ## Remove the code-assistant pod
	helm uninstall $(HELM_RELEASE) 2>/dev/null || true
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

logs: ## Stream pod logs
	kubectl logs $(POD_NAME) -f

status: ## Show pod status and resource allocation
	@kubectl get pod $(POD_NAME) 2>/dev/null || echo "Pod not running"
	@kubectl get service $(POD_NAME) 2>/dev/null || true
	@echo ""
	@echo "GPU resources:"
	@kubectl get node -o json | jq -r '.items[0].status.allocatable["gpu.intel.com/xe"] // "not available"'
	@echo ""
	@echo "Built images:"
	@$(NERDCTL) images 2>/dev/null | grep code-assistant || echo "  (none)"

smoke: ## Run smoke tests (chat + tool call) against ENDPOINT (default: http://localhost:30083)
	@ENDPOINT="$${ENDPOINT:-http://localhost:30083}"; \
		echo "Smoke testing $$ENDPOINT"; \
		bash smoke/chat.sh "$$ENDPOINT" && \
		bash smoke/tools.sh "$$ENDPOINT"

clean: ## Remove all code-assistant images
	@$(NERDCTL) images 2>/dev/null | grep code-assistant | awk '{print $$1":"$$2}' | \
		xargs -r $(NERDCTL) rmi 2>/dev/null || true
	@echo "Cleaned up images"
