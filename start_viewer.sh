python3 \
  src/qemu_memory_viewer/main.py \
    ./guest486.ram \
    --width 1024 \
    --height 1024 \
    --fps 30 \
    --qmp-sock /tmp/qmp-486.sock
