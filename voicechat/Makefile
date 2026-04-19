MODEL      ?= qwen3-8b
WHISPER    ?= base
SPEAKER    ?= ljspeech
DEVICES    ?= npu,gpu
TTS_DEVICE ?= CPU

IMAGE_BASE       := localhost/voicechat
POD_NAME         := voicechat
CONTAINERD_SOCK  := /run/k3s/containerd/containerd.sock
NERDCTL          := sudo nerdctl --address $(CONTAINERD_SOCK) --namespace k8s.io
HELM_RELEASE     := voicechat

# Parse model config (NAME|HF_MODEL_ID|MODEL_DIR)
MODEL_LINE := $(shell grep "^$(MODEL)|" models.conf)
MODEL_ID   := $(word 2,$(subst |, ,$(MODEL_LINE)))
MODEL_DIR  := $(word 3,$(subst |, ,$(MODEL_LINE)))
IMAGE_NAME := $(IMAGE_BASE):$(MODEL)

# Derive whisper model dir from WHISPER variant name
WHISPER_DIR := /models/whisper-$(WHISPER)

TTS_DIR := /models/speecht5

# Parse DEVICES into helm --set flags
HELM_DEVICE_FLAGS := --set npu.enabled=false --set gpu.enabled=false
ifneq (,$(findstring npu,$(DEVICES)))
HELM_DEVICE_FLAGS := $(subst npu.enabled=false,npu.enabled=true,$(HELM_DEVICE_FLAGS))
endif
ifneq (,$(findstring gpu,$(DEVICES)))
HELM_DEVICE_FLAGS := $(subst gpu.enabled=false,gpu.enabled=true,$(HELM_DEVICE_FLAGS))
endif

TRAINER_IMAGE    := wakeword-trainer
TRAINER_OUTPUT   := /tmp/wakeword_output
WAKEWORD_PHRASE       ?= hey athena
WAKEWORD_SAMPLES      ?= 1500
WAKEWORD_STEPS        ?= 15000

.PHONY: buildkit build build-all build-trainer train-wakeword deploy undeploy nuke logs status models voices clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

models: ## List available LLM models
	@echo "Available models (use MODEL=<name>):"
	@echo ""
	@printf "  %-16s %s\n" "NAME" "HUGGING FACE MODEL"
	@printf "  %-16s %s\n" "----" "------------------"
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir; do \
		printf "  %-16s %s\n" "$$name" "$$hf"; \
	done
	@echo ""
	@echo "Example: make build MODEL=qwen3-8b WHISPER=base"

voices: ## List available TTS voices
	@echo "Available SpeechT5 voices (use SPEAKER=<name>):"
	@echo ""
	@printf "  %-16s %s\n" "NAME" "DESCRIPTION"
	@printf "  %-16s %s\n" "----" "-----------"
	@grep -v '^#' voices.conf | grep '|' | while IFS='|' read name desc; do \
		printf "  %-16s %s\n" "$$name" "$$desc"; \
	done
	@echo ""
	@echo "Whisper variants: tiny, base, small  (use WHISPER=<variant>)"

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

build-trainer: buildkit ## Build the wake word training container
	$(NERDCTL) build -f Dockerfile.train-wakeword -t $(TRAINER_IMAGE) .

train-wakeword: ## Train a custom wake word (WAKEWORD_PHRASE="hey athena" WAKEWORD_SAMPLES=1500 WAKEWORD_STEPS=15000)
	@$(NERDCTL) images --quiet $(TRAINER_IMAGE) 2>/dev/null | grep -q . || \
		{ echo "ERROR: Training image not found. Run 'make build-trainer' first."; exit 1; }
	@mkdir -p $(TRAINER_OUTPUT)
	@echo "Training wake word: '$(WAKEWORD_PHRASE)'"
	@echo "  Samples: $(WAKEWORD_SAMPLES)  Steps: $(WAKEWORD_STEPS)"
	@echo "  Output:  $(TRAINER_OUTPUT)/"
	$(NERDCTL) run --rm \
		--shm-size 2g \
		-v $(TRAINER_OUTPUT):/output \
		$(TRAINER_IMAGE) \
		"$(WAKEWORD_PHRASE)" \
		--n-samples $(WAKEWORD_SAMPLES) \
		--steps $(WAKEWORD_STEPS)
	@echo ""
	@echo "Output files:"
	@ls -lh $(TRAINER_OUTPUT)/

build: buildkit ## Build image (MODEL=qwen3-8b WHISPER=base SPEAKER=ljspeech TTS_DEVICE=CPU)
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "ERROR: Unknown model '$(MODEL)'. Run 'make models' to see available models."; \
		exit 1; \
	fi
	@echo "Building $(IMAGE_NAME)..."
	@echo "  LLM:     $(MODEL_ID)"
	@echo "  Whisper: $(WHISPER)"
	@echo "  Speaker: $(SPEAKER) (SpeechT5)"
	@sha256sum wakewords/*.onnx > wakewords/.checksum
	$(NERDCTL) build \
		--build-arg MODEL_ID=$(MODEL_ID) \
		--build-arg MODEL_DIR=$(MODEL_DIR) \
		--build-arg WHISPER=$(WHISPER) \
		-t $(IMAGE_NAME) .

build-all: buildkit ## Build images for all models in models.conf
	@grep -v '^#' models.conf | grep '|' | while IFS='|' read name hf dir; do \
		echo ""; \
		echo "========== Building $$name =========="; \
		$(NERDCTL) build \
			--build-arg MODEL_ID=$$hf \
			--build-arg MODEL_DIR=$$dir \
			--build-arg WHISPER=$(WHISPER) \
			--build-arg SPEAKER=$(SPEAKER) \
			-t $(IMAGE_BASE):$$name . || \
		echo "WARN: $$name failed, continuing..."; \
	done

deploy: ## Deploy voice chatbot (MODEL=qwen3-8b DEVICES=npu,gpu)
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
		--set whisperDir=$(WHISPER_DIR) \
		--set speaker=$(SPEAKER) \
		--set ttsDir=$(TTS_DIR) \
		--set ttsDevice=$(TTS_DEVICE) \
		$(HELM_DEVICE_FLAGS)
	@echo ""
	@echo "Waiting for pod to start..."
	@kubectl wait --for=condition=Ready pod/$(POD_NAME) --timeout=600s 2>/dev/null || \
		(echo "Pod not ready. Check: make logs"; exit 1)
	@echo ""
	@NODE_IP=$$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'); \
	echo "============================================================"; \
	echo "  Voice Chatbot ready! ($(MODEL))"; \
	echo "  Whisper: $(WHISPER)  |  Speaker: $(SPEAKER) (SpeechT5)"; \
	echo "  Devices: $(DEVICES)"; \
	echo ""; \
	echo "  Web UI:  http://$$NODE_IP:30082"; \
	echo ""; \
	echo "  Note: Web UI unavailable until all engines finish loading."; \
	echo "        LLM compilation may take several minutes on first run."; \
	echo "        Use 'make logs' to monitor loading progress."; \
	echo "============================================================"

undeploy: ## Remove the voice chatbot pod
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
	@echo "NPU resources:"
	@kubectl get node -o json | jq -r '.items[0].status.allocatable["npu.intel.com/accel"] // "not available"'
	@echo ""
	@echo "Built images:"
	@$(NERDCTL) images 2>/dev/null | grep voicechat || echo "  (none)"

clean: ## Remove all voicechat images
	@$(NERDCTL) images 2>/dev/null | grep voicechat | awk '{print $$1":"$$2}' | \
		xargs -r $(NERDCTL) rmi 2>/dev/null || true
	@echo "Cleaned up images"
