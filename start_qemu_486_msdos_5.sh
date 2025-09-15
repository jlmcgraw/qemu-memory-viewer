
dd if=/dev/zero of=guest486.ram bs=1m count=1

# qemu-img create -f qcow cdrive.qcow 32M

#qemu-system-i386 -M isapc -cpu 486 -m 1M \
#  -object memory-backend-file,id=ram1,mem-path=/tmp/guest486.ram,share=on,size=1M \
#  -machine accel=tcg,memory-backend=ram1 \
#  -hda ./cdrive.qcow \
#  -boot d -cdrom "MS-DOS 6.22.iso"
##   -fda Dos6.22.img
##   -boot a

#qemu-system-i386 -M isapc -cpu 486 -m 1M \
#  -object memory-backend-file,id=ram1,mem-path=/tmp/guest486.ram,share=on,size=1M \
#  -machine accel=tcg,memory-backend=ram1 \
#  -qmp unix:/tmp/qmp-486.sock,server,nowait \


#qemu-system-i386 \
#  -M pc -cpu 486 -m 1M \
#  -object memory-backend-file,id=ram1,mem-path=/tmp/guest486.ram,share=on,size=1M \
#  -machine accel=tcg,memory-backend=ram1 \
#  -device isa-vga \
#  -qmp unix:/tmp/qmp-486.sock,server,nowait \
#  -boot d -cdrom "MS-DOS 6.22.iso"

qemu-system-i386 \
  -M isapc \
  -cpu 486 \
  -m 1M \
  -object memory-backend-file,id=ram1,mem-path=guest486.ram,share=on,size=1M \
  -machine accel=tcg,memory-backend=ram1 \
  -hda ./cdrive.qcow \
  -qmp unix:/tmp/qmp-486.sock,server,nowait

#  -fda  "microsoft-ms-dos-5-microsoft-1991/Microsoft MS-DOS 5 (Microsoft)(1991) - Disk 1.img" \
#  -boot a \