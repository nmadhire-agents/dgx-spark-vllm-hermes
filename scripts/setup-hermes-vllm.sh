#!/usr/bin/env bash
set -Eeuo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

MODEL_DEFAULT="nvidia/Qwen3.6-35B-A3B-NVFP4"

HF_MODEL_HANDLE="${HF_MODEL_HANDLE:-$MODEL_DEFAULT}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.23.0}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-hermes-vllm}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_CONTAINER_HOST="${VLLM_CONTAINER_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_SHM_SIZE="${VLLM_SHM_SIZE:-16g}"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/.cache/huggingface}"
HERMES_BASE_URL="${HERMES_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
HERMES_MODEL="${HERMES_MODEL:-$HF_MODEL_HANDLE}"
HERMES_INSTALL_URL="${HERMES_INSTALL_URL:-https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh}"
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
SYSTEMD_UNIT_NAME="${SYSTEMD_UNIT_NAME:-hermes-vllm.service}"

log() {
  printf '[setup-hermes-vllm] %s\n' "$*"
}

die() {
  printf '[setup-hermes-vllm] ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  have "$1" || die "Missing required command: $1"
}

docker_cmd() {
  docker "$@"
}

hermes_bin() {
  if have hermes; then
    command -v hermes
  elif [ -x "$HOME/.local/bin/hermes" ]; then
    printf '%s\n' "$HOME/.local/bin/hermes"
  else
    return 1
  fi
}

print_config() {
  cat <<EOF
HF_MODEL_HANDLE=$HF_MODEL_HANDLE
VLLM_IMAGE=$VLLM_IMAGE
VLLM_CONTAINER_NAME=$VLLM_CONTAINER_NAME
VLLM_HOST=$VLLM_HOST
VLLM_PORT=$VLLM_PORT
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS
HERMES_BASE_URL=$HERMES_BASE_URL
HERMES_MODEL=$HERMES_MODEL
EOF
}

check_prereqs() {
  require_command uname
  require_command curl
  require_command docker

  log "Host:"
  uname -a

  if have nvidia-smi; then
    log "NVIDIA GPU:"
    nvidia-smi
  else
    log "nvidia-smi not found; Docker GPU access may still work if the toolkit is installed."
  fi

  docker_cmd ps >/dev/null || die "Docker daemon is not reachable by this user. Add the user to the docker group or run with appropriate privileges."
  docker_cmd info --format '{{json .Runtimes}}' | grep -q nvidia || log "NVIDIA runtime was not listed by docker info; continuing because modern Docker may still support --gpus."

  if [ -z "${HF_TOKEN:-}" ]; then
    die "HF_TOKEN is required to download the model from Hugging Face."
  fi

  mkdir -p "$VLLM_CACHE_DIR"
  log "Prerequisites look usable."
}

pull_vllm() {
  check_prereqs
  log "Pulling vLLM image: $VLLM_IMAGE"
  docker_cmd pull "$VLLM_IMAGE"
}

remove_existing_container() {
  if docker_cmd ps -a --format '{{.Names}}' | grep -qx "$VLLM_CONTAINER_NAME"; then
    log "Removing existing container: $VLLM_CONTAINER_NAME"
    docker_cmd rm -f "$VLLM_CONTAINER_NAME" >/dev/null
  fi
}

start_vllm() {
  check_prereqs
  remove_existing_container

  log "Starting vLLM for $HF_MODEL_HANDLE on ${VLLM_HOST}:${VLLM_PORT}"
  docker_cmd run -d \
    --name "$VLLM_CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --shm-size "$VLLM_SHM_SIZE" \
    -p "${VLLM_HOST}:${VLLM_PORT}:${VLLM_PORT}" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "VLLM_USE_FLASHINFER_MOE_FP4=0" \
    -e "VLLM_FP8_MOE_BACKEND=flashinfer_cutlass" \
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1" \
    -e "CUTE_DSL_ARCH=sm_121a" \
    -v "${VLLM_CACHE_DIR}:/root/.cache/huggingface" \
    "$VLLM_IMAGE" \
    "$HF_MODEL_HANDLE" \
      --host "$VLLM_CONTAINER_HOST" \
      --port "$VLLM_PORT" \
      --tensor-parallel-size 1 \
      --trust-remote-code \
      --dtype auto \
      --quantization modelopt \
      --kv-cache-dtype fp8 \
      --attention-backend flashinfer \
      --moe-backend marlin \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
      --enable-chunked-prefill \
      --async-scheduling \
      --enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' >/dev/null

  log "Container started. Follow logs with: docker logs -f $VLLM_CONTAINER_NAME"
}

stop_vllm() {
  if docker_cmd ps -a --format '{{.Names}}' | grep -qx "$VLLM_CONTAINER_NAME"; then
    log "Stopping and removing $VLLM_CONTAINER_NAME"
    docker_cmd rm -f "$VLLM_CONTAINER_NAME" >/dev/null
  else
    log "No container named $VLLM_CONTAINER_NAME exists."
  fi
}

wait_for_vllm() {
  require_command curl
  local deadline="${1:-900}"
  local start
  start="$(date +%s)"

  log "Waiting up to ${deadline}s for vLLM at $HERMES_BASE_URL/models"
  while true; do
    if curl -fsS "${HERMES_BASE_URL}/models" >/dev/null 2>&1; then
      log "vLLM is responding."
      return 0
    fi

    if [ "$(( $(date +%s) - start ))" -ge "$deadline" ]; then
      docker_cmd logs --tail 80 "$VLLM_CONTAINER_NAME" || true
      die "Timed out waiting for vLLM."
    fi

    sleep 10
  done
}

verify_vllm() {
  require_command curl
  log "Checking vLLM models endpoint."
  curl -fsS "${HERMES_BASE_URL}/models"
  printf "\n"

  log "Sending a small chat completion test."
  sample_response
}

sample_response() {
  require_command curl
  local response
  local content

  response="$(
    curl -fsS "${HERMES_BASE_URL}/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${HERMES_MODEL}\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a concise health-check responder. Do not explain.\"},{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: Hermes can use the local vLLM model.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  )"

  content="$(printf "%s" "$response" | sed -n "s/.*\"content\":\"\([^\"]*\)\".*/\1/p" | sed "s/\\n/ /g; s/\\\"/\"/g")"
  if [ -z "$content" ]; then
    printf "%s\n" "$response"
    die "vLLM chat completion did not include a sample response."
  fi

  log "Sample response from $HERMES_MODEL:"
  printf "%s\n" "$content"
}

install_hermes() {
  require_command curl
  if hermes_bin >/dev/null 2>&1; then
    log "Hermes already installed at $(hermes_bin)"
    return 0
  fi

  log "Installing Hermes Agent."
  curl -fsSL "$HERMES_INSTALL_URL" | bash
}

configure_hermes() {
  local bin
  bin="$(hermes_bin)" || die "Hermes is not installed. Run: $0 install-hermes"

  log "Configuring Hermes custom endpoint: $HERMES_BASE_URL"
  "$bin" config set model.provider custom
  "$bin" config set model.base_url "$HERMES_BASE_URL"
  "$bin" config set model.default "$HERMES_MODEL"
}

verify_hermes() {
  local bin
  bin="$(hermes_bin)" || die "Hermes is not installed. Run: $0 install-hermes"
  log "Hermes installed at $bin"
  log "Verifying the configured Hermes model endpoint with a bounded sample response."
  sample_response
}

install_vllm_service() {
  check_prereqs
  mkdir -p "$SYSTEMD_USER_DIR"

  local env_file
  env_file="$SYSTEMD_USER_DIR/hermes-vllm.env"

  log "Writing $env_file"
  cat > "$env_file" <<EOF
HF_TOKEN=$HF_TOKEN
HF_MODEL_HANDLE=$HF_MODEL_HANDLE
VLLM_IMAGE=$VLLM_IMAGE
VLLM_CONTAINER_NAME=$VLLM_CONTAINER_NAME
VLLM_HOST=$VLLM_HOST
VLLM_CONTAINER_HOST=$VLLM_CONTAINER_HOST
VLLM_PORT=$VLLM_PORT
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS
VLLM_SHM_SIZE=$VLLM_SHM_SIZE
VLLM_CACHE_DIR=$VLLM_CACHE_DIR
EOF
  chmod 600 "$env_file"

  log "Writing $SYSTEMD_USER_DIR/$SYSTEMD_UNIT_NAME"
  cat > "$SYSTEMD_USER_DIR/$SYSTEMD_UNIT_NAME" <<EOF
[Unit]
Description=vLLM for Hermes Agent
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
EnvironmentFile=$env_file
ExecStart=$PWD/scripts/setup-hermes-vllm.sh start-vllm
ExecStop=$PWD/scripts/setup-hermes-vllm.sh stop-vllm
TimeoutStartSec=0

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable "$SYSTEMD_UNIT_NAME"
  systemctl --user restart "$SYSTEMD_UNIT_NAME"
  log "Installed and started user service: $SYSTEMD_UNIT_NAME"
}

install_all() {
  print_config
  pull_vllm
  start_vllm
  wait_for_vllm
  install_hermes
  configure_hermes
  verify_hermes
  log "Setup complete. Run 'hermes' to open the TUI or 'hermes gateway setup' for Telegram."
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  print-config          Print effective configuration
  check-prereqs         Check host, Docker, GPU, and HF_TOKEN
  pull-vllm             Pull the configured vLLM image
  start-vllm            Start vLLM Docker container
  stop-vllm             Stop and remove vLLM Docker container
  wait-for-vllm [secs]  Wait for vLLM API readiness
  verify-vllm           Test vLLM OpenAI-compatible API
  install-hermes        Install Hermes Agent if missing
  configure-hermes      Configure Hermes to use vLLM
  verify-hermes         Verify Hermes config and print a vLLM sample response
  sample-response       Print a bounded sample response from configured vLLM
  install-vllm-service  Install/start a user systemd service for vLLM
  install-all           Pull vLLM, start server, install/configure/test Hermes
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    print-config) print_config ;;
    check-prereqs) check_prereqs ;;
    pull-vllm) pull_vllm ;;
    start-vllm) start_vllm ;;
    stop-vllm) stop_vllm ;;
    wait-for-vllm) wait_for_vllm "${2:-900}" ;;
    verify-vllm) verify_vllm ;;
    sample-response) sample_response ;;
    install-hermes) install_hermes ;;
    configure-hermes) configure_hermes ;;
    verify-hermes) verify_hermes ;;
    install-vllm-service) install_vllm_service ;;
    install-all) install_all ;;
    -h|--help|help|"") usage ;;
    *) usage; die "Unknown command: $command" ;;
  esac
}

main "$@"
