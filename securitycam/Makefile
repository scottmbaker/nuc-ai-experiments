DEVICES ?= npu,gpu
MODE ?= test
IMAGE_NAME := localhost/securitycam:latest
POD_NAME := securitycam
CONTAINERD_SOCK := /run/k3s/containerd/containerd.sock
NERDCTL := nerdctl --address $(CONTAINERD_SOCK) --namespace k8s.io
HELM_RELEASE := securitycam

# Parse DEVICES into helm --set flags
HELM_DEVICE_FLAGS := --set npu.enabled=false --set gpu.enabled=false
ifneq (,$(findstring npu,$(DEVICES)))
HELM_DEVICE_FLAGS := $(subst npu.enabled=false,npu.enabled=true,$(HELM_DEVICE_FLAGS))
endif
ifneq (,$(findstring gpu,$(DEVICES)))
HELM_DEVICE_FLAGS := $(subst gpu.enabled=false,gpu.enabled=true,$(HELM_DEVICE_FLAGS))
endif

.PHONY: buildkit build cameras deploy undeploy nuke logs status clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

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

build: buildkit ## Build the securitycam image
	@echo "Building $(IMAGE_NAME)..."
	$(NERDCTL) build -t $(IMAGE_NAME) .

cameras: ## Create/update k8s secret from cameras.yaml
	@if [ ! -f cameras.yaml ]; then \
		echo "ERROR: cameras.yaml not found."; \
		echo "  cp cameras.yaml.example cameras.yaml"; \
		echo "  # edit with your real camera URLs"; \
		exit 1; \
	fi
	kubectl create secret generic securitycam-cameras \
		--from-file=cameras.yaml=cameras.yaml \
		--dry-run=client -o yaml | kubectl apply -f -
	@echo "Secret securitycam-cameras updated"

deploy: ## Deploy to k3s (DEVICES=npu,gpu MODE=test|live)
	@echo "Deploying securitycam with devices: $(DEVICES)"
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
		--set image=$(IMAGE_NAME) \
		--set mode=$(MODE) \
		$(HELM_DEVICE_FLAGS)
	@echo ""
	@echo "Waiting for pod to start..."
	@kubectl wait --for=condition=Ready pod/$(POD_NAME) --timeout=300s 2>/dev/null || \
		(echo "Pod not ready. Check: make logs"; exit 1)
	@echo ""
	@NODE_IP=$$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'); \
	echo "============================================================"; \
	echo "  Security Camera AI is ready!"; \
	echo "  Mode: $(MODE)  Devices: $(DEVICES)"; \
	echo ""; \
	echo "    http://$$NODE_IP:30083"; \
	echo ""; \
	echo "============================================================"

undeploy: ## Remove the deployment
	helm uninstall $(HELM_RELEASE) 2>/dev/null || true

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

logs: ## Show pod logs
	kubectl logs $(POD_NAME) -f

status: ## Show pod status
	@kubectl get pod $(POD_NAME) 2>/dev/null || echo "Pod not running"
	@kubectl get service $(POD_NAME) 2>/dev/null || true
	@echo ""
	@echo "Built images:"
	@$(NERDCTL) images 2>/dev/null | grep securitycam || echo "  (none)"

clean: ## Remove all securitycam images
	@$(NERDCTL) images 2>/dev/null | grep securitycam | awk '{print $$1":"$$2}' | \
		xargs -r $(NERDCTL) rmi 2>/dev/null || true
	@echo "Cleaned up images"
