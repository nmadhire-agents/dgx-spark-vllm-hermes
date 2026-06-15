#!/usr/bin/env bash
set -Eeuo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

MODEL_DEFAULT="nvidia/Qwen3.6-35B-A3B-NVFP4"

HF_MODEL_HANDLE="${HF_MODEL_HANDLE:-$MODEL_DEFAULT}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
OPENCODE_PROVIDER_ID="${OPENCODE_PROVIDER_ID:-vllm}"
OPENCODE_PROVIDER_NAME="${OPENCODE_PROVIDER_NAME:-vLLM (local)}"
OPENCODE_MODEL_NAME="${OPENCODE_MODEL_NAME:-$HF_MODEL_HANDLE (local)}"
OPENCODE_CONFIG_PATH="${OPENCODE_CONFIG_PATH:-opencode.json}"
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
OPENCODE_MODEL="${OPENCODE_MODEL:-$HF_MODEL_HANDLE}"
OPENCODE_MODEL_REF="${OPENCODE_MODEL_REF:-${OPENCODE_PROVIDER_ID}/${OPENCODE_MODEL}}"
SETUP_HERMES_VLLM_SCRIPT="${SETUP_HERMES_VLLM_SCRIPT:-./scripts/setup-hermes-vllm.sh}"

log() {
  printf '[setup-opencode-vllm] %s\n' "$*"
}

die() {
  printf '[setup-opencode-vllm] ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  have "$1" || die "Missing required command: $1"
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

opencode_bin() {
  if have opencode; then
    command -v opencode
  elif [ -x "$HOME/.opencode/bin/opencode" ]; then
    printf '%s\n' "$HOME/.opencode/bin/opencode"
  else
    return 1
  fi
}

require_opencode() {
  opencode_bin >/dev/null 2>&1 || die "OpenCode is not installed or is not on PATH."
}

require_vllm_script() {
  [ -x "$SETUP_HERMES_VLLM_SCRIPT" ] || die "Missing executable vLLM setup script: $SETUP_HERMES_VLLM_SCRIPT"
}

print_config() {
  cat <<EOF
HF_MODEL_HANDLE=$HF_MODEL_HANDLE
VLLM_HOST=$VLLM_HOST
VLLM_PORT=$VLLM_PORT
OPENCODE_PROVIDER_ID=$OPENCODE_PROVIDER_ID
OPENCODE_PROVIDER_NAME=$OPENCODE_PROVIDER_NAME
OPENCODE_CONFIG_PATH=$OPENCODE_CONFIG_PATH
OPENCODE_BASE_URL=$OPENCODE_BASE_URL
OPENCODE_MODEL=$OPENCODE_MODEL
OPENCODE_MODEL_REF=$OPENCODE_MODEL_REF
SETUP_HERMES_VLLM_SCRIPT=$SETUP_HERMES_VLLM_SCRIPT
EOF
}

check_prereqs() {
  require_command curl
  require_opencode
  require_vllm_script
  log "OpenCode installed at $(opencode_bin)"
  log "vLLM setup script found at $SETUP_HERMES_VLLM_SCRIPT"
}

write_opencode_config() {
  local config_path="$OPENCODE_CONFIG_PATH"
  local provider_id provider_name base_url model model_name model_ref tmp_path

  provider_id="$(json_escape "$OPENCODE_PROVIDER_ID")"
  provider_name="$(json_escape "$OPENCODE_PROVIDER_NAME")"
  base_url="$(json_escape "$OPENCODE_BASE_URL")"
  model="$(json_escape "$OPENCODE_MODEL")"
  model_name="$(json_escape "$OPENCODE_MODEL_NAME")"
  model_ref="$(json_escape "$OPENCODE_MODEL_REF")"
  tmp_path="${config_path}.tmp"

  if [ -e "$config_path" ] && [ "${OPENCODE_OVERWRITE_CONFIG:-0}" != "1" ]; then
    die "$config_path already exists. Set OPENCODE_OVERWRITE_CONFIG=1 to replace it, or set OPENCODE_CONFIG_PATH to a different file."
  fi

  log "Writing OpenCode config: $config_path"
  cat > "$tmp_path" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "$provider_id": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "$provider_name",
      "options": {
        "baseURL": "$base_url"
      },
      "models": {
        "$model": {
          "name": "$model_name",
          "limit": {
            "context": $VLLM_MAX_MODEL_LEN,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "$model_ref",
  "small_model": "$model_ref"
}
EOF
  mv "$tmp_path" "$config_path"
}

start_vllm() {
  require_vllm_script
  "$SETUP_HERMES_VLLM_SCRIPT" start-vllm
}

stop_vllm() {
  require_vllm_script
  "$SETUP_HERMES_VLLM_SCRIPT" stop-vllm
}

wait_for_vllm() {
  require_vllm_script
  "$SETUP_HERMES_VLLM_SCRIPT" wait-for-vllm "${1:-900}"
}

verify_vllm() {
  require_command curl
  log "Checking vLLM models endpoint."
  curl -fsS "${OPENCODE_BASE_URL}/models"
  printf '\n'
}

verify_opencode_config() {
  local bin
  bin="$(opencode_bin)" || die "OpenCode is not installed or is not on PATH."
  log "Resolved OpenCode config:"
  "$bin" debug config
}

verify_opencode_run() {
  local bin
  bin="$(opencode_bin)" || die "OpenCode is not installed or is not on PATH."
  log "Running a bounded OpenCode prompt against $OPENCODE_MODEL_REF"
  "$bin" run --model "$OPENCODE_MODEL_REF" "Reply exactly OPENCODE_VLLM_OK"
}

install_all() {
  print_config
  check_prereqs
  start_vllm
  wait_for_vllm
  write_opencode_config
  verify_vllm
  log "Setup complete. Run 'opencode' in this repo, or 'opencode run --model \"$OPENCODE_MODEL_REF\" \"Reply exactly OPENCODE_VLLM_OK\"'."
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  print-config             Print effective configuration
  check-prereqs            Check OpenCode, curl, and the vLLM setup script
  configure-opencode       Write OpenCode project config for local vLLM
  start-vllm               Start vLLM through setup-hermes-vllm.sh
  stop-vllm                Stop vLLM through setup-hermes-vllm.sh
  wait-for-vllm [secs]     Wait for vLLM API readiness
  verify-vllm              Test the vLLM OpenAI-compatible API
  verify-opencode-config   Print OpenCode's resolved config
  verify-opencode-run      Send a small OpenCode CLI prompt to local vLLM
  install-all              Start vLLM, wait, configure OpenCode, and verify vLLM

Environment overrides:
  OPENCODE_CONFIG_PATH=$OPENCODE_CONFIG_PATH
  OPENCODE_OVERWRITE_CONFIG=1
  OPENCODE_PROVIDER_ID=$OPENCODE_PROVIDER_ID
  OPENCODE_MODEL=$OPENCODE_MODEL
  OPENCODE_BASE_URL=$OPENCODE_BASE_URL
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    print-config) print_config ;;
    check-prereqs) check_prereqs ;;
    configure-opencode) check_prereqs; write_opencode_config ;;
    start-vllm) start_vllm ;;
    stop-vllm) stop_vllm ;;
    wait-for-vllm) wait_for_vllm "${2:-900}" ;;
    verify-vllm) verify_vllm ;;
    verify-opencode-config) verify_opencode_config ;;
    verify-opencode-run) verify_opencode_run ;;
    install-all) install_all ;;
    -h|--help|help|"") usage ;;
    *) usage; die "Unknown command: $command" ;;
  esac
}

main "$@"
