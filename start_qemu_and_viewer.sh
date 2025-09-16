#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QEMU_BIN=${QEMU_BINARY:-qemu-system-i386}
RAM_FILE="${SCRIPT_DIR}/guest486.ram"
QMP_SOCKET=${QMP_SOCKET:-/tmp/qmp-486.sock}
VIEWER_SCRIPT="${SCRIPT_DIR}/src/qemu_memory_viewer/main.py"
VIEWER_WIDTH=${VIEWER_WIDTH:-1024}
VIEWER_HEIGHT=${VIEWER_HEIGHT:-1024}
VIEWER_FPS=${VIEWER_FPS:-30}
DEFAULT_DISK_IMAGE="${SCRIPT_DIR}/cdrive.qcow"
QEMU_DISK_IMAGE=${QEMU_DISK_IMAGE:-${DEFAULT_DISK_IMAGE}}

cleanup() {
  if [[ -n "${QEMU_PID:-}" ]]; then
    if kill -0 "${QEMU_PID}" 2>/dev/null; then
      echo "Stopping QEMU (pid ${QEMU_PID})..."
      kill "${QEMU_PID}" 2>/dev/null || true
      wait "${QEMU_PID}" 2>/dev/null || true
    fi
  fi
  if [[ -S "${QMP_SOCKET}" ]]; then
    rm -f "${QMP_SOCKET}"
  fi
}

on_interrupt() {
  echo
  echo "Interrupt received, shutting down..." >&2
  exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

command -v "${QEMU_BIN}" >/dev/null 2>&1 || {
  echo "Error: QEMU binary '${QEMU_BIN}' not found." >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || {
  echo "Error: python3 is required to start the viewer." >&2
  exit 1
}

if [[ ! -f "${VIEWER_SCRIPT}" ]]; then
  echo "Error: viewer script not found at ${VIEWER_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${QEMU_DISK_IMAGE}" ]]; then
  if [[ ! -f "${QEMU_DISK_IMAGE}" ]]; then
    cat <<MSG >&2
Error: QEMU disk image '${QEMU_DISK_IMAGE}' not found.
       Create the image or set QEMU_DISK_IMAGE="" to start without a drive.
MSG
    exit 1
  fi
fi

if [[ -e "${QMP_SOCKET}" ]]; then
  echo "Removing stale QMP socket at ${QMP_SOCKET}"
  rm -f "${QMP_SOCKET}"
fi

echo "Creating RAM backing file at ${RAM_FILE}"
dd if=/dev/zero of="${RAM_FILE}" bs=1m count=1 status=none

QEMU_ARGS=(
  -M isapc
  -cpu 486
  -m 1M
  -object "memory-backend-file,id=ram1,mem-path=${RAM_FILE},share=on,size=1M"
  -machine "accel=tcg,memory-backend=ram1"
  -qmp "unix:${QMP_SOCKET},server,nowait"
)

if [[ -n "${QEMU_DISK_IMAGE}" ]]; then
  QEMU_ARGS+=( -hda "${QEMU_DISK_IMAGE}" )
fi

echo "Starting QEMU using ${QEMU_BIN}"
"${QEMU_BIN}" "${QEMU_ARGS[@]}" &
QEMU_PID=$!
echo "QEMU started with PID ${QEMU_PID}"

# Wait for the QMP socket to become available so the viewer can connect.
for _ in $(seq 1 50); do
  if [[ -S "${QMP_SOCKET}" ]]; then
    break
  fi
  if ! kill -0 "${QEMU_PID}" 2>/dev/null; then
    echo "QEMU exited unexpectedly while starting." >&2
    wait "${QEMU_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 0.1
done

if [[ ! -S "${QMP_SOCKET}" ]]; then
  echo "Warning: QMP socket ${QMP_SOCKET} not detected; the viewer will retry on connect." >&2
fi

echo "Launching qemu-memory-viewer"
python3 "${VIEWER_SCRIPT}" \
  "${RAM_FILE}" \
  --width "${VIEWER_WIDTH}" \
  --height "${VIEWER_HEIGHT}" \
  --fps "${VIEWER_FPS}" \
  --qmp-sock "${QMP_SOCKET}"

# If the viewer exits normally we still want to ensure QEMU is shut down.
if [[ -n "${QEMU_PID:-}" ]]; then
  wait "${QEMU_PID}" 2>/dev/null || true
fi
