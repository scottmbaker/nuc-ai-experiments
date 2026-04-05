MODEL ?= lcm-dreamshaper
DEVICES ?= npu
TEST_MODELS ?= lcm-dreamshaper sd15 sdxl-turbo
TEST_PROMPT ?= a red sports car in a showroom
IMAGE_BASE := localhost/npu-imagegen
POD_NAME := npu-imagegen
CONTAINERD_SOCK := /run/k3s/containerd/containerd.sock
NERDCTL := nerdctl --address $(CONTAINERD_SOCK) --namespace k8s.io
HELM_RELEASE := npu-imagegen

# Parse model config
MODEL_LINE := $(shell grep "^$(MODEL)|" models.conf)
MODEL_ID := $(word 2,$(subst |, ,$(MODEL_LINE)))
MODEL_DIR := $(word 3,$(subst |, ,$(MODEL_LINE)))
PIPELINE_CLASS := $(word 4,$(subst |, ,$(MODEL_LINE)))
DOWNLOAD_METHOD := $(word 5,$(subst |, ,$(MODEL_LINE)))
DEFAULT_STEPS := $(word 6,$(subst |, ,$(MODEL_LINE)))
DEFAULT_GUIDANCE := $(word 7,$(subst |, ,$(MODEL_LINE)))
IMAGE_NAME := $(IMAGE_BASE):$(MODEL)

# Parse DEVICES into helm --set flags
HELM_DEVICE_FLAGS := --set npu.enabled=false --set gpu.enabled=false
ifneq (,$(findstring npu,$(DEVICES)))
HELM_DEVICE_FLAGS := $(subst npu.enabled=false,npu.enabled=true,$(HELM_DEVICE_FLAGS))
endif
ifneq (,$(findstring gpu,$(DEVICES)))
HELM_DEVICE_FLAGS := $(subst gpu.enabled=false,gpu.enabled=true,$(HELM_DEVICE_FLAGS))
endif

.PHONY: buildkit build build-all deploy undeploy nuke models logs status clean test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

models: ## List available models
	@echo "Available models (use MODEL=<name>):"
	@echo ""
	@printf "  %-20s %-50s %s\n" "NAME" "HUGGING FACE MODEL" "METHOD"
	@printf "  %-20s %-50s %s\n" "----" "------------------" "------"
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir pipe method steps guidance; do \
		printf "  %-20s %-50s %s\n" "$$name" "$$hf" "$$method"; \
	done
	@echo ""
	@echo "Note: sdxl-base and playground-v25 may produce unpredictable results"
	@echo "      (image grids, abstract output) due to INT8 quantization artifacts."
	@echo ""
	@echo "Example: make build MODEL=sd15"
	@echo "         make deploy MODEL=sd15 DEVICES=npu,gpu"

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
	@echo "  Method: $(DOWNLOAD_METHOD)"
	$(NERDCTL) build \
		--build-arg MODEL_ID=$(MODEL_ID) \
		--build-arg MODEL_DIR=$(MODEL_DIR) \
		--build-arg PIPELINE_CLASS=$(PIPELINE_CLASS) \
		--build-arg DOWNLOAD_METHOD=$(DOWNLOAD_METHOD) \
		--build-arg DEFAULT_STEPS=$(DEFAULT_STEPS) \
		--build-arg DEFAULT_GUIDANCE=$(DEFAULT_GUIDANCE) \
		-t $(IMAGE_NAME) .

build-all: buildkit ## Build images for all models
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir pipe method steps guidance; do \
		echo ""; \
		echo "========== Building $$name =========="; \
		echo "  Model: $$hf"; \
		$(NERDCTL) build \
			--build-arg MODEL_ID=$$hf \
			--build-arg MODEL_DIR=$$dir \
			--build-arg PIPELINE_CLASS=$$pipe \
			--build-arg DOWNLOAD_METHOD=$$method \
			--build-arg DEFAULT_STEPS=$$steps \
			--build-arg DEFAULT_GUIDANCE=$$guidance \
			-t $(IMAGE_BASE):$$name . || \
		echo "WARN: $$name failed to build, continuing..."; \
	done
	@echo ""
	@echo "All builds complete."

deploy: ## Deploy a model (MODEL=<name>, DEVICES=npu,gpu,cpu)
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "ERROR: Unknown model '$(MODEL)'. Run 'make models' to see available models."; \
		exit 1; \
	fi
	@echo "Deploying $(MODEL) with devices: $(DEVICES)"
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
		--set modelDir=$(MODEL_DIR) \
		$(HELM_DEVICE_FLAGS)
	@echo ""
	@echo "Waiting for pod to start..."
	@kubectl wait --for=condition=Ready pod/$(POD_NAME) --timeout=300s 2>/dev/null || \
		(echo "Pod not ready. Check: make logs"; exit 1)
	@echo ""
	@NODE_IP=$$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'); \
	echo "============================================================"; \
	echo "  Image Generator is ready! ($(MODEL))"; \
	echo "  Devices: $(DEVICES)"; \
	echo ""; \
	echo "    http://$$NODE_IP:30080"; \
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
	@$(NERDCTL) images 2>/dev/null | grep npu-imagegen || echo "  (none)"

test: ## Test all models on all devices (TEST_MODELS=... TEST_PROMPT=...)
	@echo "============================================================"
	@echo "  Image Generation Test"
	@echo "  Prompt: $(TEST_PROMPT)"
	@echo "============================================================"
	@printf "\n%-20s %-6s %-10s %s\n" "MODEL" "DEVICE" "TIME" "STATUS"
	@printf "%-20s %-6s %-10s %s\n" "-----" "------" "----" "------"
	@for model in $(TEST_MODELS); do \
		line=$$(grep "^$$model|" models.conf); \
		dir=$$(echo "$$line" | cut -d'|' -f3); \
		steps=$$(echo "$$line" | cut -d'|' -f6); \
		guidance=$$(echo "$$line" | cut -d'|' -f7); \
		echo ""; \
		echo "--- Building $$model ---"; \
		sudo make -s build MODEL=$$model 2>/dev/null; \
		echo "--- Deploying $$model ---"; \
		kubectl delete pod $(POD_NAME) --force --grace-period=0 2>/dev/null; \
		make -s nuke 2>/dev/null; \
		sleep 3; \
		make -s deploy MODEL=$$model DEVICES=npu,gpu 2>/dev/null; \
		echo "Waiting for model to load..."; \
		for i in $$(seq 1 120); do \
			ready=$$(kubectl exec $(POD_NAME) -- python3 -c "import urllib.request,json; r=urllib.request.urlopen('http://localhost:8080/status'); print(json.loads(r.read()).get('ready',''))" 2>/dev/null); \
			if [ "$$ready" = "True" ]; then break; fi; \
			sleep 5; \
		done; \
		if [ "$$ready" != "True" ]; then \
			printf "%-20s %-6s %-10s %s\n" "$$model" "-" "-" "LOAD FAILED"; \
			continue; \
		fi; \
		for device in CPU GPU NPU; do \
			test_result=$$(kubectl exec $(POD_NAME) -- python3 -W ignore -c "import urllib.request,json,base64,io;from PIL import Image;req=urllib.request.Request('http://localhost:8080/generate',data=json.dumps({'prompt':'$(TEST_PROMPT)','device':'$$device','steps':$$steps,'guidance_scale':$$guidance,'seed':42}).encode(),headers={'Content-Type':'application/json'});r=urllib.request.urlopen(req,timeout=600);data=json.loads(r.read());img=base64.b64decode(data['image_base64']);im=Image.open(io.BytesIO(img));pixels=list(im.getdata());avg=sum(sum(p[:3]) for p in pixels[:1000])/(1000*3);elapsed=data['elapsed_seconds'];print(f'BLACK|{elapsed}' if avg<1.0 else f'OK|{elapsed}')" 2>&1); \
			status=$$(echo "$$test_result" | grep -oE '^(OK|BLACK|ERROR)' || echo "ERROR"); \
			elapsed=$$(echo "$$test_result" | cut -d'|' -f2); \
			if [ "$$status" = "OK" ]; then \
				printf "%-20s %-6s %-10s %s\n" "$$model" "$$device" "$${elapsed}s" "OK"; \
			elif [ "$$status" = "BLACK" ]; then \
				printf "%-20s %-6s %-10s %s\n" "$$model" "$$device" "$${elapsed}s" "FAIL: black image"; \
			else \
				printf "%-20s %-6s %-10s %s\n" "$$model" "$$device" "-" "FAIL: $$test_result"; \
			fi; \
		done; \
	done
	@echo ""
	@echo "============================================================"
	@echo "  Test complete"
	@echo "============================================================"

clean: ## Remove all npu-imagegen images
	@$(NERDCTL) images 2>/dev/null | grep npu-imagegen | awk '{print $$1":"$$2}' | \
		xargs -r $(NERDCTL) rmi 2>/dev/null || true
	@echo "Cleaned up images"
