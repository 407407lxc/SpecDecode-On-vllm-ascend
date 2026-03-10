#!/usr/bin/env bash
set -euo pipefail

# Show NPU process info and clear only after explicit confirmation.

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[ERROR] npu-smi not found in PATH." >&2
  exit 1
fi

output="$(npu-smi info)"
printf '%s\n' "$output"

mapfile -t pids < <(
  printf '%s\n' "$output" |
    awk -F'|' '
      /^[|]/ {
        pid = $3
        gsub(/^[ \t]+|[ \t]+$/, "", pid)
        if (pid ~ /^[0-9]+$/) {
          print pid
        }
      }
    ' |
    sort -u
)

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "[INFO] No running NPU process found."
  exit 0
fi

echo ""
echo "[INFO] Detected NPU process IDs: ${pids[*]}"
echo "[INFO] Process details:"
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    ps -fp "$pid" || true
  else
    echo "[WARN] PID $pid is not alive anymore."
  fi
done

echo ""
read -r -p "Clear these processes now? Type yes/no: " confirm
if [[ "$confirm" != "yes" ]]; then
  echo "[INFO] Abort cleanup."
  exit 0
fi

echo "[INFO] Sending SIGTERM..."
for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

sleep 2

still_alive=()
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    still_alive+=("$pid")
  fi
done

if [[ ${#still_alive[@]} -gt 0 ]]; then
  echo "[WARN] Still alive after SIGTERM: ${still_alive[*]}"
  read -r -p "Force kill with SIGKILL? Type yes/no: " force
  if [[ "$force" == "yes" ]]; then
    for pid in "${still_alive[@]}"; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    echo "[INFO] SIGKILL sent."
  else
    echo "[INFO] Keep remaining processes alive."
  fi
else
  echo "[INFO] Cleanup finished with SIGTERM."
fi