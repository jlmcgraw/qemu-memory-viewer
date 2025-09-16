#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "matplotlib", "Pillow"]
# ///

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised when numpy is available
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback exercised in tests
    from . import _compat_numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from PIL import ImageFont

PANEL_SIZE = 64  # 64x64 “under cursor” byte window
VGA_ROWS, VGA_COLS = 25, 80
VGA_TEXT_BASE = 0xB8000
VGA_TEXT_BYTES = VGA_ROWS * VGA_COLS * 2  # char+attr

# 16-color VGA palette (approx)
VGA_RGB = [
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
]

# CP437 LUT
_CP437: List[str] = []
for b in range(256):
    ch = bytes([b]).decode("cp437", errors="replace")
    if (0x00 <= b <= 0x1F) or b == 0x7F:
        ch = "."
    _CP437.append(ch)

def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def compute_marker_rows(markers_kib: List[int], full_width: int, full_height: int) -> List[int]:
    rows = []
    bpr = float(full_width)
    for kib in markers_kib:
        r = int((kib * 1024) // bpr)
        if 0 <= r < full_height:
            rows.append(r)
    return rows


def pick_mono_font(size: int = 13) -> "ImageFont.FreeTypeFont":
    try:
        from matplotlib import font_manager as fm
        from PIL import ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when optional deps missing
        msg = "Font rendering requires both Pillow and Matplotlib"
        raise RuntimeError(msg) from exc

    path = fm.findfont("DejaVu Sans Mono", fallback_to_default=True)
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:  # pragma: no cover - Pillow fallback path
        return ImageFont.load_default()


def bytes_to_cp437_lines(block: np.ndarray) -> List[str]:
    lines: List[str] = []
    for y in range(PANEL_SIZE):
        row = block[y, :PANEL_SIZE]
        lines.append("".join(_CP437[int(v)] for v in row))
    return lines


def render_text_panel(block: np.ndarray, font: "ImageFont.FreeTypeFont") -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when Pillow missing
        raise RuntimeError("Rendering text panels requires Pillow") from exc

    lines = bytes_to_cp437_lines(block)
    try:
        l, t, r, b = font.getbbox("M")
        cell_w = max(8, r - l)
        ascent, descent = font.getmetrics()
        cell_h = max(12, ascent + descent)
    except Exception:
        cell_w, cell_h = 8, 12
    img = Image.new("L", (PANEL_SIZE * cell_w, PANEL_SIZE * cell_h), color=255)
    draw = ImageDraw.Draw(img)
    y = 0
    for text in lines:
        draw.text((0, y), text, fill=0, font=font)
        y += cell_h
    return np.asarray(img, dtype=np.uint8)


# -------- QMP minimal client --------

class QMP:
    def __init__(self, path: str):
        self.path = path
        self.sock: Optional[socket.socket] = None
        self.buf = b""

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.path)
        self.sock = s
        self._recv_json()  # greeting
        self._send_json({"execute": "qmp_capabilities"})
        self._recv_json()

    def _send_json(self, obj: dict) -> None:
        assert self.sock is not None
        data = (json.dumps(obj) + "\r\n").encode("utf-8")
        self.sock.sendall(data)

    def _recv_json(self) -> dict:
        assert self.sock is not None
        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("QMP socket closed")
            self.buf += chunk
            while b"\r\n" in self.buf:
                line, self.buf = self.buf.split(b"\r\n", 1)
                if not line.strip():
                    continue
                return json.loads(line.decode("utf-8"))

    def hmp(self, cmd: str) -> str:
        self._send_json({"execute": "human-monitor-command", "arguments": {"command-line": cmd}})
        resp = self._recv_json()
        if "return" in resp:
            return str(resp["return"])
        raise RuntimeError(f"HMP error: {resp}")


HEX_BYTE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{2})(?![0-9a-fA-F])")
REG_TOKEN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([0-9A-Fa-f]+)")
SEGMENT_LINE = re.compile(
    r"^(?P<seg>[A-Za-z]{2})\s*=\s*(?P<selector>[0-9A-Fa-f]{1,4})\s+(?P<base>[0-9A-Fa-f]{8,16})"
)


@dataclass(frozen=True)
class RegisterPointerSpec:
    """Description of a register-backed pointer to plot on the memory map."""

    label: str
    offset_keys: Sequence[str]
    segment: Optional[str] = None
    color: str = "cyan"


POINTER_SPECS: Tuple[RegisterPointerSpec, ...] = (
    RegisterPointerSpec(label="IP", offset_keys=("RIP", "EIP", "IP"), segment="CS", color="#00d7ff"),
)


def _extract_hex_bytes(text: str, limit: int) -> List[int]:
    vals: List[int] = []
    for match in HEX_BYTE.finditer(text):
        vals.append(int(match.group(1), 16))
        if len(vals) >= limit:
            break
    return vals


def parse_register_dump(text: str) -> Dict[str, int]:
    registers: Dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for name, value in REG_TOKEN.findall(stripped):
            registers[name.upper()] = int(value, 16)
        seg_match = SEGMENT_LINE.match(stripped)
        if seg_match:
            seg = seg_match.group("seg").upper()
            selector = int(seg_match.group("selector"), 16)
            base = int(seg_match.group("base"), 16)
            registers.setdefault(seg, selector)
            registers[f"{seg}_BASE"] = base
    return registers


def compute_pointer_address(registers: Dict[str, int], spec: RegisterPointerSpec) -> Optional[int]:
    offset: Optional[int] = None
    for key in spec.offset_keys:
        key_upper = key.upper()
        if key_upper in registers:
            offset = registers[key_upper]
            break
    if offset is None:
        return None

    if spec.segment is None:
        return offset

    seg_name = spec.segment.upper()
    base = registers.get(f"{seg_name}_BASE")
    if base is None and seg_name in registers:
        base = registers[seg_name] << 4
    if base is None:
        return None
    return base + offset


def qmp_read_b800(q: QMP) -> np.ndarray:
    txt = q.hmp(f"xp /{VGA_TEXT_BYTES}bx {VGA_TEXT_BASE}")
    vals = _extract_hex_bytes(txt, VGA_TEXT_BYTES)
    if len(vals) < VGA_TEXT_BYTES:
        vals += [0] * (VGA_TEXT_BYTES - len(vals))
    arr = np.frombuffer(bytearray(vals[:VGA_TEXT_BYTES]), dtype=np.uint8)
    return arr.reshape(VGA_ROWS, VGA_COLS, 2)


def render_vga_text(chars_attrs: np.ndarray, font: "ImageFont.FreeTypeFont") -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when Pillow missing
        raise RuntimeError("Rendering VGA panels requires Pillow") from exc

    try:
        l, t, r, b = font.getbbox("M")
        cw = max(8, r - l)
        ascent, descent = font.getmetrics()
        ch = max(12, ascent + descent)
    except Exception:
        cw, ch = 8, 12
    img = Image.new("RGB", (VGA_COLS * cw, VGA_ROWS * ch), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(VGA_ROWS):
        for x in range(VGA_COLS):
            c = int(chars_attrs[y, x, 0])
            a = int(chars_attrs[y, x, 1])
            fg = VGA_RGB[a & 0x0F]
            bg = VGA_RGB[(a >> 4) & 0x07]
            draw.rectangle([x * cw, y * ch, x * cw + cw, y * ch + ch], fill=bg)
            draw.text((x * cw, y * ch), _CP437[c], fill=fg, font=font)
    return np.asarray(img, dtype=np.uint8)


# -------- Main viewer --------

def main() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as exc:  # pragma: no cover - viewer path only
        raise RuntimeError("Matplotlib is required to run the viewer") from exc

    p = argparse.ArgumentParser(description="Live memory viewer with CP437 magnifier and VGA B800h panel")
    p.add_argument("path", help="RAM file path, e.g., /tmp/guest486.ram")
    p.add_argument("--width", type=int, default=1024, help="bytes per row in RAM view")
    p.add_argument("--height", type=int, default=1024, help="rows in RAM view")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--min-window", type=int, default=16)
    p.add_argument("--qmp-sock", required=True, help="QMP UNIX socket path (e.g., /tmp/qmp-486.sock)")
    args = p.parse_args()

    size = os.path.getsize(args.path)
    pixels = args.width * args.height
    if size < pixels:
        raise SystemExit(f"file too small: {size} bytes < {pixels} required")

    mm = np.memmap(args.path, dtype=np.uint8, mode="r")
    full = mm[:pixels].reshape(args.height, args.width)

    vx0, vy0 = 0, 0
    vW, vH = args.width, args.height
    need_axes_refresh = True
    guide_lines: list = []

    sel_cx: Optional[int] = args.width // 2
    sel_cy: Optional[int] = args.height // 2

    markers_kib = [0, 128, 256, 512, 640, 768]
    marker_rows_full = compute_marker_rows(markers_kib, args.width, args.height)

    def view_slice() -> np.ndarray:
        return full[vy0:vy0 + vH, vx0:vx0 + vW]

    def refresh_axes() -> None:
        nonlocal guide_lines, need_axes_refresh
        for ln in guide_lines:
            ln.remove()
        guide_lines.clear()
        ax.set_xlim(-0.5, vW - 0.5)
        ax.set_ylim(vH - 0.5, -0.5)
        y_ticks: List[int] = []
        y_labels: List[str] = []
        for kib, row in zip(markers_kib, marker_rows_full):
            if vy0 <= row < (vy0 + vH):
                y = row - vy0
                y_ticks.append(y)
                y_labels.append(f"{kib} KiB")
                guide_lines.append(ax.axhline(y - 0.5, linewidth=0.6, color="0.7"))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        need_axes_refresh = False

    def set_view(center: Tuple[int, int], scale: float) -> None:
        nonlocal vx0, vy0, vW, vH, need_axes_refresh
        cx, cy = center
        newW = clamp(int(round(vW * scale)), args.min_window, args.width)
        newH = clamp(int(round(vH * scale)), args.min_window, args.height)
        vx = int(round(cx - newW / 2))
        vy = int(round(cy - newH / 2))
        vx0 = clamp(vx, 0, args.width - newW)
        vy0 = clamp(vy, 0, args.height - newH)
        vW, vH = newW, newH
        need_axes_refresh = True

    def pan(dx: int, dy: int) -> None:
        nonlocal vx0, vy0, need_axes_refresh
        vx0 = clamp(vx0 + dx, 0, args.width - vW)
        vy0 = clamp(vy0 + dy, 0, args.height - vH)
        need_axes_refresh = True

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[3, 2, 2])
    ax = fig.add_subplot(gs[0, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_vga = fig.add_subplot(gs[0, 2])
    ax_mag.set_title(f"{PANEL_SIZE}×{PANEL_SIZE} CP437 bytes")
    ax_mag.set_axis_off()
    ax_vga.set_title("VGA text @ B800:0000 (80×25)")
    ax_vga.set_axis_off()

    im = ax.imshow(view_slice(), cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper")
    ax.set_title(f"{args.path} ({args.width}x{args.height})")
    ax.margins(0)
    rect = patches.Rectangle((0, 0), PANEL_SIZE, PANEL_SIZE, linewidth=1.2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)

    pointer_artists: List[Tuple[RegisterPointerSpec, object, object]] = []
    for spec in POINTER_SPECS:
        marker_line, = ax.plot(
            [], [],
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=spec.color,
            markeredgewidth=1.5,
            linestyle="none",
        )
        label_text = ax.text(
            0,
            0,
            spec.label,
            color=spec.color,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox=dict(facecolor=(0.0, 0.0, 0.0, 0.6), edgecolor="none", pad=1.5),
        )
        marker_line.set_visible(False)
        label_text.set_visible(False)
        pointer_artists.append((spec, marker_line, label_text))

    font_small = pick_mono_font(13)
    font_vga = pick_mono_font(14)

    sx0 = clamp(sel_cx - PANEL_SIZE // 2, 0, args.width - PANEL_SIZE)
    sy0 = clamp(sel_cy - PANEL_SIZE // 2, 0, args.height - PANEL_SIZE)
    block0 = full[sy0:sy0 + PANEL_SIZE, sx0:sx0 + PANEL_SIZE]
    im_mag = ax_mag.imshow(
        render_text_panel(block0, font_small),
        cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper"
    )

    qmp = QMP(args.qmp_sock)
    qmp.connect()
    try:
        vga0 = qmp_read_b800(qmp)
    except Exception:
        vga0 = np.zeros((VGA_ROWS, VGA_COLS, 2), dtype=np.uint8)
    im_vga = ax_vga.imshow(render_vga_text(vga0, font_vga), interpolation="nearest", origin="upper")

    refresh_axes()

    register_state: Dict[str, int] = {}

    def update_pointer_plot(registers: Dict[str, int]) -> None:
        for spec, marker_line, label_text in pointer_artists:
            addr = compute_pointer_address(registers, spec) if registers else None
            if addr is None or not (0 <= addr < pixels):
                marker_line.set_visible(False)
                label_text.set_visible(False)
                continue

            gx = addr % args.width
            gy = addr // args.width
            if not (vx0 <= gx < vx0 + vW and vy0 <= gy < vy0 + vH):
                marker_line.set_visible(False)
                label_text.set_visible(False)
                continue

            lx = gx - vx0
            ly = gy - vy0
            marker_line.set_data([lx], [ly])
            marker_line.set_visible(True)

            label_x = float(clamp(int(round(lx + 2)), 0, max(0, vW - 1)))
            label_y = float(clamp(int(round(ly + 2)), 0, max(0, vH - 1)))
            label_text.set_position((label_x, label_y))
            label_text.set_visible(True)

    try:
        register_state = parse_register_dump(qmp.hmp("info registers"))
    except Exception:
        pass
    update_pointer_plot(register_state)

    def on_key(event):
        nonlocal vx0, vy0, vW, vH, need_axes_refresh, sel_cx, sel_cy
        if event.key in ("+", "=", "kp_add"):
            cx, cy = vx0 + vW // 2, vy0 + vH // 2
            set_view((cx, cy), 0.5)
        elif event.key in ("-", "_", "kp_subtract"):
            cx, cy = vx0 + vW // 2, vy0 + vH // 2
            set_view((cx, cy), 2.0)
        elif event.key == "left":
            pan(-(max(1, vW // 10)), 0)
        elif event.key == "right":
            pan(max(1, vW // 10), 0)
        elif event.key == "up":
            pan(0, -(max(1, vH // 10)))
        elif event.key == "down":
            pan(0, max(1, vH // 10))
        elif event.key == "0":
            vx0, vy0, vW, vH = 0, 0, args.width, args.height
            need_axes_refresh = True
        elif event.key == "m":
            sel_cx, sel_cy = vx0 + vW // 2, vy0 + vH // 2

    def on_move(event):
        nonlocal sel_cx, sel_cy
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        sel_cx = vx0 + int(round(event.xdata))
        sel_cy = vy0 + int(round(event.ydata))

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        cx = vx0 + int(round(event.xdata))
        cy = vy0 + int(round(event.ydata))
        if event.button == "up":
            set_view((cx, cy), 0.8)
        elif event.button == "down":
            set_view((cx, cy), 1.25)

    cid_k = fig.canvas.mpl_connect("key_press_event", on_key)
    cid_s = fig.canvas.mpl_connect("scroll_event", on_scroll)
    cid_m = fig.canvas.mpl_connect("motion_notify_event", on_move)

    def update(_):
        nonlocal need_axes_refresh, register_state
        im.set_data(view_slice())
        if need_axes_refresh:
            refresh_axes()

        cx = clamp(sel_cx if sel_cx is not None else args.width // 2, 0, args.width - 1)
        cy = clamp(sel_cy if sel_cy is not None else args.height // 2, 0, args.height - 1)
        sx = clamp(cx - PANEL_SIZE // 2, 0, args.width - PANEL_SIZE)
        sy = clamp(cy - PANEL_SIZE // 2, 0, args.height - PANEL_SIZE)

        rect.set_xy((sx - vx0, sy - vy0))
        rect.set_width(PANEL_SIZE)
        rect.set_height(PANEL_SIZE)

        im_mag.set_data(render_text_panel(full[sy:sy + PANEL_SIZE, sx:sx + PANEL_SIZE], font_small))

        try:
            vga = qmp_read_b800(qmp)
            im_vga.set_data(render_vga_text(vga, font_vga))
        except Exception:
            pass

        try:
            reg_txt = qmp.hmp("info registers")
        except Exception:
            pass
        else:
            register_state = parse_register_dump(reg_txt)

        update_pointer_plot(register_state)

        artists = [im, im_mag, im_vga, rect]
        for _, marker_line, label_text in pointer_artists:
            artists.extend((marker_line, label_text))
        return tuple(artists)

    interval_ms = int(1000 / max(1, args.fps))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    plt.show()
    _ = (cid_k, cid_s, cid_m, anim)


if __name__ == "__main__":
    main()
