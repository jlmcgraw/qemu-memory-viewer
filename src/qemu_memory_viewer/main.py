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
MAPPING_FETCH_CHUNK = 256
MAX_MAPPING_FETCH_BYTES = 512 * 1024

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
MAP_RANGE = re.compile(r"(?:0x)?([0-9A-Fa-f]+)\s*-\s*(?:0x)?([0-9A-Fa-f]+)")
ALIAS_TOKEN = re.compile(r"alias(?:\s+\w+)?\s*=\s*([^@]+)")


@dataclass(frozen=True)
class RegisterPointerSpec:
    """Description of a register-backed pointer to plot on the memory map."""

    label: str
    offset_keys: Sequence[str]
    segment: Optional[str] = None
    color: str = "cyan"


@dataclass(frozen=True)
class MemoryMapping:
    """Description of a memory range reported by QEMU."""

    start: int
    end: int
    label: str

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class DisplayRegion:
    """Predefined PC memory area to highlight in the viewer."""

    start: int
    end: int
    label: str
    color: str


DISPLAY_REGIONS: Tuple[DisplayRegion, ...] = (
    DisplayRegion(0x00000000, 0x000003FF, "Interrupt Vector Table", "#f4cccc"),
    DisplayRegion(0x00000400, 0x000004FF, "BDA (BIOS Data Area)", "#fce5cd"),
    DisplayRegion(0x00000500, 0x00007BFF, "Conventional memory (usable)", "#fff2cc"),
    DisplayRegion(0x00007C00, 0x00007DFF, "OS BootSector", "#d9ead3"),
    DisplayRegion(0x00007E00, 0x0007FFFF, "Conventional memory", "#cfe2f3"),
    DisplayRegion(0x00080000, 0x0009FFFF, "EBDA (Extended BIOS Data Area)", "#d9d2e9"),
    DisplayRegion(0x000A0000, 0x000BFFFF, "Video display memory", "#ead1dc"),
    DisplayRegion(0x000C0000, 0x000C7FFF, "Video BIOS", "#ffe599"),
    DisplayRegion(0x000C8000, 0x000EFFFF, "BIOS expansions", "#c9daf8"),
    DisplayRegion(0x000F0000, 0x000FFFFF, "Motherboard BIOS", "#d0e0e3"),
)


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


def parse_memory_mappings(text: str) -> List[MemoryMapping]:
    mappings: List[MemoryMapping] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("address-space"):
            continue
        range_match = MAP_RANGE.search(line)
        if not range_match:
            continue
        start_hex, end_hex = range_match.groups()
        try:
            start = int(start_hex, 16)
            end = int(end_hex, 16)
        except ValueError:
            continue
        if end < start:
            start, end = end, start

        rest = line[range_match.end():].strip()
        if rest.startswith("("):
            depth = 0
            idx = 0
            for idx, ch in enumerate(rest):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
            rest = rest[idx:].strip()
        if rest.startswith(":"):
            rest = rest[1:].strip()
        alias_match = ALIAS_TOKEN.search(rest)
        label = alias_match.group(1).strip() if alias_match else rest
        if not label:
            label = "mapping"
        if "@" in label:
            label = label.split("@", 1)[0].strip()
        label = label.strip("\"'")
        mappings.append(MemoryMapping(start=start, end=end, label=label or "mapping"))
    return mappings


def build_mapping_mask(mappings: Sequence[MemoryMapping], total_bytes: int) -> Tuple[np.ndarray, List[MemoryMapping]]:
    normalized_total = max(0, int(total_bytes))
    mask_list = [0] * normalized_total
    visible: List[MemoryMapping] = []
    if normalized_total <= 0:
        return np.asarray(mask_list, dtype=np.uint8), visible

    max_index = normalized_total - 1
    idx = 1
    for mapping in mappings:
        start = max(0, int(mapping.start))
        end = min(int(mapping.end), max_index)
        if end < start or start > max_index or end < 0:
            continue
        if idx >= 255:
            break
        span = end - start + 1
        mask_list[start:end + 1] = [idx] * span
        visible.append(mapping)
        idx += 1
    return np.asarray(mask_list, dtype=np.uint8), visible


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


def qmp_read_bytes(q: QMP, start: int, count: int, chunk_size: int = MAPPING_FETCH_CHUNK) -> np.ndarray:
    """Read a span of physical memory bytes via the QMP human monitor."""

    normalized_count = max(0, int(count))
    if normalized_count == 0:
        return np.zeros(0, dtype=np.uint8)

    chunk_len = max(1, int(chunk_size))
    base = int(start)
    buf = bytearray(normalized_count)
    offset = 0
    while offset < normalized_count:
        step = min(chunk_len, normalized_count - offset)
        addr = base + offset
        txt = q.hmp(f"xp /{step}bx 0x{addr:X}")
        vals = _extract_hex_bytes(txt, step)
        if len(vals) < step:
            vals.extend([0] * (step - len(vals)))
        buf[offset : offset + step] = bytes(vals[:step])
        offset += step
    return np.frombuffer(buf, dtype=np.uint8)


def build_mapping_overlay_data(
    qmp: QMP, mappings: Sequence[MemoryMapping], total_bytes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw bytes for selected mappings.

    Returns a tuple of ``(data, mask)`` flattened to ``total_bytes`` entries. The mask
    indicates which indices have valid overlay data.
    """

    normalized_total = max(0, int(total_bytes))
    data_list = [0] * normalized_total
    mask_list = [0] * normalized_total
    if normalized_total == 0:
        return np.asarray(data_list, dtype=np.uint8), np.asarray(mask_list, dtype=np.uint8)

    for mapping in mappings:
        start = max(0, int(mapping.start))
        end = min(int(mapping.end), normalized_total - 1)
        if end < start:
            continue

        span = end - start + 1
        if span <= 0:
            continue
        label_lc = mapping.label.lower()
        if span > MAX_MAPPING_FETCH_BYTES:
            continue
        if "system" in label_lc and "ram" in label_lc:
            continue

        try:
            chunk = qmp_read_bytes(qmp, start, span, chunk_size=MAPPING_FETCH_CHUNK)
        except Exception:
            continue

        if hasattr(chunk, "tolist"):
            chunk_vals = chunk.tolist()
        else:
            chunk_vals = list(chunk)
        if len(chunk_vals) < span:
            chunk_vals.extend([0] * (span - len(chunk_vals)))
        elif len(chunk_vals) > span:
            chunk_vals = chunk_vals[:span]

        data_list[start : end + 1] = chunk_vals
        mask_list[start : end + 1] = [1] * span

    data = np.asarray(data_list, dtype=np.uint8)
    mask = np.asarray(mask_list, dtype=np.uint8)
    return data, mask


def apply_overlay_block(
    base_block: np.ndarray,
    overlay_block: Optional[np.ndarray],
    overlay_mask_block: Optional[np.ndarray],
) -> np.ndarray:
    """Return ``base_block`` with overlay bytes applied where available."""

    base_arr = np.asarray(base_block)
    if overlay_block is None or overlay_mask_block is None:
        return base_arr

    overlay_arr = np.asarray(overlay_block)
    mask_arr = np.asarray(overlay_mask_block)

    if base_arr.shape != overlay_arr.shape:
        try:
            overlay_arr = overlay_arr.reshape(base_arr.shape)
        except Exception:
            return base_arr
    if base_arr.shape != mask_arr.shape:
        try:
            mask_arr = mask_arr.reshape(base_arr.shape)
        except Exception:
            return base_arr

    def mask_any(obj: object) -> bool:
        try:
            length = len(obj)  # type: ignore[arg-type]
        except Exception:
            return bool(obj)
        for idx in range(length):
            if mask_any(obj[idx]):  # type: ignore[index]
                return True
        return False

    if not mask_any(mask_arr):
        return base_arr

    def composite(base_obj: object, overlay_obj: object, mask_obj: object) -> object:
        try:
            length = len(base_obj)  # type: ignore[arg-type]
        except Exception:
            return int(overlay_obj) if bool(mask_obj) else int(base_obj)

        return [
            composite(base_obj[idx], overlay_obj[idx], mask_obj[idx])  # type: ignore[index]
            for idx in range(length)
        ]

    combined = composite(base_arr, overlay_arr, mask_arr)
    return np.asarray(combined, dtype=np.uint8)


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
        from matplotlib import colors as mcolors
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

    mapping_mask_full: Optional[np.ndarray] = None
    mapping_overlay_im = None
    mapping_legend = None
    mapping_visible: List[MemoryMapping] = []
    mapping_data_full: Optional[np.ndarray] = None
    mapping_data_mask_full: Optional[np.ndarray] = None

    vx0, vy0 = 0, 0
    vW, vH = args.width, args.height
    need_axes_refresh = True
    guide_lines: list = []

    sel_cx: Optional[int] = args.width // 2
    sel_cy: Optional[int] = args.height // 2

    active_regions = [region for region in DISPLAY_REGIONS if region.start < pixels]

    def format_kib_from_bytes(value: int) -> str:
        kib = value / 1024.0
        kib_str = f"{kib:.2f}".rstrip("0").rstrip(".")
        return f"{kib_str} KiB"

    marker_specs = []
    for region in active_regions:
        y_pos = region.start / float(args.width)
        if 0 <= y_pos < args.height:
            label = f"0x{region.start:08X} ({format_kib_from_bytes(region.start)})"
            marker_specs.append((y_pos, label))
    marker_specs.sort(key=lambda item: item[0])

    overlay_mappings = [
        MemoryMapping(start=region.start, end=region.end, label=region.label)
        for region in active_regions
    ]
    region_color_lookup = {
        (region.start, region.end, region.label): region.color for region in active_regions
    }

    def find_region_for_address(addr: int) -> Optional[DisplayRegion]:
        for region in active_regions:
            if region.start <= addr <= region.end:
                return region
        return None

    def view_slice() -> np.ndarray:
        return full[vy0:vy0 + vH, vx0:vx0 + vW]

    def view_slice_with_overlay() -> np.ndarray:
        base_view = view_slice()
        if mapping_data_full is None or mapping_data_mask_full is None:
            return base_view

        overlay_view = mapping_data_full[vy0:vy0 + vH, vx0:vx0 + vW]
        mask_view = mapping_data_mask_full[vy0:vy0 + vH, vx0:vx0 + vW]
        return apply_overlay_block(base_view, overlay_view, mask_view)

    def magnifier_block_at(sx: int, sy: int) -> np.ndarray:
        base_block = full[sy : sy + PANEL_SIZE, sx : sx + PANEL_SIZE]
        overlay_block = (
            mapping_data_full[sy : sy + PANEL_SIZE, sx : sx + PANEL_SIZE]
            if mapping_data_full is not None
            else None
        )
        mask_block = (
            mapping_data_mask_full[sy : sy + PANEL_SIZE, sx : sx + PANEL_SIZE]
            if mapping_data_mask_full is not None
            else None
        )
        return apply_overlay_block(base_block, overlay_block, mask_block)

    def refresh_axes() -> None:
        nonlocal guide_lines, need_axes_refresh
        for ln in guide_lines:
            ln.remove()
        guide_lines.clear()
        ax.set_xlim(-0.5, vW - 0.5)
        ax.set_ylim(vH - 0.5, -0.5)
        y_ticks: List[float] = []
        y_labels: List[str] = []
        for y_pos, label in marker_specs:
            if vy0 <= y_pos < (vy0 + vH):
                y = y_pos - vy0
                y_ticks.append(y)
                y_labels.append(label)
                guide_lines.append(ax.axhline(y - 0.5, linewidth=0.6, color="0.7"))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis="y", which="both", labelsize=8, pad=4)
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

    im = ax.imshow(
        view_slice_with_overlay(),
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        origin="upper",
    )
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

    hover_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor=(0.0, 0.0, 0.0, 0.65), edgecolor="none", pad=1.8),
    )

    font_small = pick_mono_font(13)
    font_vga = pick_mono_font(14)

    sx0 = clamp(sel_cx - PANEL_SIZE // 2, 0, args.width - PANEL_SIZE)
    sy0 = clamp(sel_cy - PANEL_SIZE // 2, 0, args.height - PANEL_SIZE)
    block0 = magnifier_block_at(sx0, sy0)
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

    if overlay_mappings:
        mask_flat, mapping_visible = build_mapping_mask(overlay_mappings, pixels)
        if mapping_visible:
            mapping_mask_full = mask_flat.reshape(args.height, args.width)

            color_entries = [(0.0, 0.0, 0.0, 0.0)]
            for mapping in mapping_visible:
                color = region_color_lookup.get((mapping.start, mapping.end, mapping.label), "#cccccc")
                rgba = mcolors.to_rgba(color, alpha=0.25)
                color_entries.append(rgba)
            overlay_cmap = mcolors.ListedColormap(color_entries, name="memory_layout")
            overlay_norm = mcolors.BoundaryNorm(np.arange(len(color_entries) + 1) - 0.5, len(color_entries))
            mapping_overlay_im = ax.imshow(
                mapping_mask_full[vy0:vy0 + vH, vx0:vx0 + vW],
                cmap=overlay_cmap,
                norm=overlay_norm,
                interpolation="nearest",
                origin="upper",
                zorder=im.get_zorder() + 0.1,
            )
            legend_handles: List[patches.Patch] = []
            for idx, mapping in enumerate(mapping_visible, start=1):
                rgba = overlay_cmap(idx)
                handle = patches.Patch(
                    facecolor=rgba,
                    edgecolor=(rgba[0], rgba[1], rgba[2], min(1.0, rgba[3] + 0.2)),
                    label=f"{mapping.label} (0x{mapping.start:08X}-0x{mapping.end:08X})",
                )
                legend_handles.append(handle)
            if legend_handles:
                mapping_legend = ax.legend(
                    handles=legend_handles,
                    loc="upper left",
                    fontsize=8,
                    framealpha=0.65,
                    title="Memory layout",
                )
                mapping_legend.set_zorder(mapping_overlay_im.get_zorder() + 0.1)

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
        im.set_data(view_slice_with_overlay())
        if mapping_overlay_im is not None and mapping_mask_full is not None:
            mapping_overlay_im.set_data(mapping_mask_full[vy0:vy0 + vH, vx0:vx0 + vW])
            mapping_overlay_im.set_extent((-0.5, vW - 0.5, vH - 0.5, -0.5))
        if need_axes_refresh:
            refresh_axes()

        cx = clamp(sel_cx if sel_cx is not None else args.width // 2, 0, args.width - 1)
        cy = clamp(sel_cy if sel_cy is not None else args.height // 2, 0, args.height - 1)
        sx = clamp(cx - PANEL_SIZE // 2, 0, args.width - PANEL_SIZE)
        sy = clamp(cy - PANEL_SIZE // 2, 0, args.height - PANEL_SIZE)

        rect.set_xy((sx - vx0, sy - vy0))
        rect.set_width(PANEL_SIZE)
        rect.set_height(PANEL_SIZE)

        im_mag.set_data(render_text_panel(magnifier_block_at(sx, sy), font_small))

        addr = cy * args.width + cx
        region = find_region_for_address(addr)
        if region is None:
            hover_lines = ["Unlabeled memory", f"0x{addr:08X} ({format_kib_from_bytes(addr)})"]
        else:
            hover_lines = [region.label, f"0x{addr:08X} ({format_kib_from_bytes(addr)})"]
        hover_text.set_text("\n".join(hover_lines))

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

        artists = [im]
        if mapping_overlay_im is not None:
            artists.append(mapping_overlay_im)
        artists.extend([im_mag, im_vga, rect, hover_text])
        for _, marker_line, label_text in pointer_artists:
            artists.extend((marker_line, label_text))
        return tuple(artists)

    interval_ms = int(1000 / max(1, args.fps))
    anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    plt.show()
    _ = (cid_k, cid_s, cid_m, anim)


if __name__ == "__main__":
    main()
