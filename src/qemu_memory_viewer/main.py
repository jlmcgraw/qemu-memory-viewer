#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "matplotlib", "Pillow"]
# ///

"""Interactive viewer for QEMU guest memory with textual and VGA overlays."""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import collections.abc as cabc
import json
import logging
import os
import re
import socket
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

if __package__ in (None, ""):
    import importlib

    PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    np = importlib.import_module("qemu_memory_viewer._compat_numpy")
else:  # pragma: no cover - exercised via unit tests
    from . import _compat_numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from numpy import ndarray as NDArray
    from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
    from PIL import ImageFont

    FreeTypeFont = ImageFont.FreeTypeFont
else:
    NDArray: TypeAlias = Any
    FreeTypeFont: TypeAlias = Any

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
_CP437: list[str] = []
for b in range(256):
    ch = bytes([b]).decode("cp437", errors="replace")
    if (0x00 <= b <= 0x1F) or b == 0x7F:
        ch = "."
    _CP437.append(ch)

def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp ``v`` to the inclusive ``[lo, hi]`` range."""
    return lo if v < lo else hi if v > hi else v


def _is_sequence(obj: object) -> bool:
    """Return ``True`` for non-string sequences."""
    return isinstance(obj, cabc.Sequence) and not isinstance(
        obj,
        (str, bytes, bytearray),
    )


def pick_mono_font(size: int = 13) -> FreeTypeFont:
    """Return a readable monospace font, falling back to Pillow's default."""
    try:
        from matplotlib import font_manager as fm
        from PIL import ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover
        msg = "Font rendering requires both Pillow and Matplotlib"
        raise RuntimeError(msg) from exc

    path = fm.findfont("DejaVu Sans Mono", fallback_to_default=True)
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:  # pragma: no cover - Pillow fallback path
        return ImageFont.load_default()


def bytes_to_cp437_lines(block: NDArray) -> list[str]:
    """Decode ``block`` into CP437 text lines suitable for the magnifier."""
    lines: list[str] = []
    for y in range(PANEL_SIZE):
        row = block[y, :PANEL_SIZE]
        lines.append("".join(_CP437[int(v)] for v in row))
    return lines


def render_text_panel(block: NDArray, font: FreeTypeFont) -> NDArray:
    """Render a CP437 character panel for the magnifier view."""
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Rendering text panels requires Pillow") from exc

    lines = bytes_to_cp437_lines(block)
    try:
        left, _top, right, _bottom = font.getbbox("M")
        cell_w = max(8, right - left)
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


class QMPClient(Protocol):
    """Protocol describing the subset of QMP used by helper functions."""

    def hmp(self, cmd: str) -> str:
        """Execute a human-monitor command and return its response."""


class QMP(QMPClient):
    """Tiny helper that speaks the subset of QMP the viewer requires."""

    def __init__(self, path: str) -> None:
        """Store the UNIX domain socket path for later connections."""
        self.path = path
        self.sock: socket.socket | None = None
        self.buf = b""

    def connect(self) -> None:
        """Establish the QMP connection and negotiate capabilities."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.path)
        self.sock = s
        self._recv_json()  # greeting
        self._send_json({"execute": "qmp_capabilities"})
        self._recv_json()

    def _send_json(self, obj: dict[str, object]) -> None:
        if self.sock is None:
            msg = "QMP socket has not been connected"
            raise RuntimeError(msg)
        data = (json.dumps(obj) + "\r\n").encode("utf-8")
        self.sock.sendall(data)

    def _recv_json(self) -> dict[str, object]:
        if self.sock is None:
            msg = "QMP socket has not been connected"
            raise RuntimeError(msg)
        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("QMP socket closed")
            self.buf += chunk
            while b"\r\n" in self.buf:
                line, self.buf = self.buf.split(b"\r\n", 1)
                if not line.strip():
                    continue
                return cast("dict[str, object]", json.loads(line.decode("utf-8")))

    def hmp(self, cmd: str) -> str:
        """Execute a human-monitor command and return its textual result."""
        self._send_json(
            {
                "execute": "human-monitor-command",
                "arguments": {"command-line": cmd},
            }
        )
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
    offset_keys: cabc.Sequence[str]
    segment: str | None = None
    color: str = "cyan"


@dataclass(frozen=True)
class MemoryMapping:
    """Description of a memory range reported by QEMU."""

    start: int
    end: int
    label: str

    @property
    def size(self) -> int:
        """Return the inclusive length of the mapping."""
        return self.end - self.start + 1


@dataclass(frozen=True)
class DisplayRegion:
    """Predefined PC memory area to highlight in the viewer."""

    start: int
    end: int
    label: str
    color: str


DISPLAY_REGIONS: tuple[DisplayRegion, ...] = (
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


POINTER_SPECS: tuple[RegisterPointerSpec, ...] = (
    RegisterPointerSpec(
        label="IP",
        offset_keys=("RIP", "EIP", "IP"),
        segment="CS",
        color="#00d7ff",
    ),
)


def _extract_hex_bytes(text: str, limit: int) -> list[int]:
    """Return up to ``limit`` hexadecimal byte values parsed from ``text``."""
    vals: list[int] = []
    for match in HEX_BYTE.finditer(text):
        vals.append(int(match.group(1), 16))
        if len(vals) >= limit:
            break
    return vals


def parse_register_dump(text: str) -> dict[str, int]:
    """Extract register values from QEMU's ``info registers`` output."""
    registers: dict[str, int] = {}
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


def parse_memory_mappings(text: str) -> list[MemoryMapping]:
    """Parse ``info mem`` output into structured mapping descriptors."""
    mappings: list[MemoryMapping] = []
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


def build_mapping_mask(
    mappings: cabc.Sequence[MemoryMapping], total_bytes: int,
) -> tuple[NDArray, list[MemoryMapping]]:
    """Create an index mask describing which mapping covers each byte."""
    normalized_total = max(0, int(total_bytes))
    mask_list = [0] * normalized_total
    visible: list[MemoryMapping] = []
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


def compute_pointer_address(
    registers: dict[str, int], spec: RegisterPointerSpec,
) -> int | None:
    """Resolve ``spec`` against ``registers`` to produce an absolute address."""
    offset: int | None = None
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


def qmp_read_b800(q: QMPClient) -> NDArray:
    """Fetch the VGA text memory as a ``(rows, cols, 2)`` uint8 array."""
    txt = q.hmp(f"xp /{VGA_TEXT_BYTES}bx {VGA_TEXT_BASE}")
    vals = _extract_hex_bytes(txt, VGA_TEXT_BYTES)
    if len(vals) < VGA_TEXT_BYTES:
        vals += [0] * (VGA_TEXT_BYTES - len(vals))
    arr = np.frombuffer(bytearray(vals[:VGA_TEXT_BYTES]), dtype=np.uint8)
    return arr.reshape(VGA_ROWS, VGA_COLS, 2)


def qmp_read_bytes(
    q: QMPClient, start: int, count: int, chunk_size: int = MAPPING_FETCH_CHUNK,
) -> NDArray:
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
    qmp: QMPClient, mappings: cabc.Sequence[MemoryMapping], total_bytes: int,
) -> tuple[NDArray, NDArray]:
    """Collect raw bytes for selected mappings.

    Returns a tuple of ``(data, mask)`` flattened to ``total_bytes`` entries. The mask
    indicates which indices have valid overlay data.
    """
    normalized_total = max(0, int(total_bytes))
    data_list = [0] * normalized_total
    mask_list = [0] * normalized_total
    if normalized_total == 0:
        return (
            np.asarray(data_list, dtype=np.uint8),
            np.asarray(mask_list, dtype=np.uint8),
        )

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
        except Exception as exc:  # pragma: no cover - overlay data is optional
            logger.debug(
                "Failed to fetch overlay chunk 0x%X-0x%X: %s", start, end, exc
            )
            continue

        chunk_vals = chunk.tolist() if hasattr(chunk, "tolist") else list(chunk)
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
    base_block: NDArray,
    overlay_block: NDArray | None,
    overlay_mask_block: NDArray | None,
) -> NDArray:
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
        return any(mask_any(obj[idx]) for idx in range(length))  # type: ignore[index]

    if not mask_any(mask_arr):
        return base_arr

    def composite(base_obj: object, overlay_obj: object, mask_obj: object) -> object:
        try:
            length = len(base_obj)  # type: ignore[arg-type]
        except Exception:
            overlay_val = cast("int", overlay_obj)
            base_val = cast("int", base_obj)
            return overlay_val if bool(mask_obj) else base_val

        return [
            composite(base_obj[idx], overlay_obj[idx], mask_obj[idx])  # type: ignore[index]
            for idx in range(length)
        ]

    combined = composite(base_arr, overlay_arr, mask_arr)
    return np.asarray(combined, dtype=np.uint8)


def render_vga_text(
    chars_attrs: NDArray, font: FreeTypeFont,
) -> NDArray:
    """Render the VGA text buffer into an RGB image."""
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Rendering VGA panels requires Pillow") from exc

    try:
        left, _top, right, _bottom = font.getbbox("M")
        cw = max(8, right - left)
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
    """Launch the interactive matplotlib memory viewer."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib import patches
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as exc:  # pragma: no cover - viewer path only
        raise RuntimeError("Matplotlib is required to run the viewer") from exc

    p = argparse.ArgumentParser(
        description="Live memory viewer with CP437 magnifier and VGA B800h panel",
    )
    p.add_argument("path", help="RAM file path, e.g., /tmp/guest486.ram")
    p.add_argument("--width", type=int, default=1024, help="bytes per row in RAM view")
    p.add_argument("--height", type=int, default=1024, help="rows in RAM view")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--min-window", type=int, default=16)
    p.add_argument(
        "--qmp-sock",
        required=True,
        help="QMP UNIX socket path (e.g., /tmp/qmp-486.sock)",
    )
    args = p.parse_args()

    size = os.path.getsize(args.path)
    pixels = args.width * args.height
    if size < pixels:
        raise SystemExit(f"file too small: {size} bytes < {pixels} required")

    mm = np.memmap(args.path, dtype=np.uint8, mode="r")
    full = mm[:pixels].reshape(args.height, args.width)

    mapping_mask_full: NDArray | None = None
    mapping_overlay_im = None
    mapping_visible: list[MemoryMapping] = []
    mapping_data_full: NDArray | None = None
    mapping_data_mask_full: NDArray | None = None

    vx0, vy0 = 0, 0
    vW, vH = args.width, args.height
    need_axes_refresh = True
    guide_lines: list[Any] = []

    sel_cx = args.width // 2
    sel_cy = args.height // 2

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
        (region.start, region.end, region.label): region.color
        for region in active_regions
    }

    def find_region_for_address(addr: int) -> DisplayRegion | None:
        for region in active_regions:
            if region.start <= addr <= region.end:
                return region
        return None

    def view_slice() -> NDArray:
        return full[vy0:vy0 + vH, vx0:vx0 + vW]

    def view_slice_with_overlay() -> NDArray:
        base_view = view_slice()
        if mapping_data_full is None or mapping_data_mask_full is None:
            return base_view

        overlay_view = mapping_data_full[vy0:vy0 + vH, vx0:vx0 + vW]
        mask_view = mapping_data_mask_full[vy0:vy0 + vH, vx0:vx0 + vW]
        return apply_overlay_block(base_view, overlay_view, mask_view)

    def flatten_array(arr: NDArray | None) -> list[int]:
        if arr is None:
            return []

        source = arr
        if hasattr(source, "tolist"):
            try:
                source = source.tolist()
            except TypeError:
                source = arr

        flat: list[int] = []
        stack = [source]
        while stack:
            item = stack.pop()
            if _is_sequence(item):
                stack.extend(reversed(item))
                continue
            try:
                flat.append(int(item))
            except Exception:
                flat.append(int(bool(item)))

        flat.reverse()
        return flat

    def mask_has_values(arr: NDArray | None) -> bool:
        return any(bool(v) for v in flatten_array(arr))

    def magnifier_block_at(sx: int, sy: int) -> NDArray:
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
        y_ticks: list[float] = []
        y_labels: list[str] = []
        for y_pos, label in marker_specs:
            if vy0 <= y_pos < (vy0 + vH):
                y = y_pos - vy0
                y_ticks.append(y)
                y_labels.append(label)
                guide_lines.append(ax.axhline(y - 0.5, linewidth=0.6, color="0.7"))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis="y", which="both", labelsize=8, pad=4)
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )
        need_axes_refresh = False

    def set_view(center: tuple[int, int], scale: float) -> None:
        nonlocal vx0, vy0, vW, vH, need_axes_refresh
        cx, cy = center
        newW = clamp(round(vW * scale), args.min_window, args.width)
        newH = clamp(round(vH * scale), args.min_window, args.height)
        vx = round(cx - newW / 2)
        vy = round(cy - newH / 2)
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
    gs = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[3, 1.2, 2, 2])
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_mag = fig.add_subplot(gs[0, 2])
    ax_vga = fig.add_subplot(gs[0, 3])
    ax_legend.set_axis_off()
    ax_mag.set_title(f"{PANEL_SIZE}x{PANEL_SIZE} CP437 bytes")
    ax_mag.set_axis_off()
    ax_vga.set_title("VGA text @ B800:0000 (80x25)")
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
    rect = patches.Rectangle(
        (0, 0),
        PANEL_SIZE,
        PANEL_SIZE,
        linewidth=1.2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)

    pointer_artists: list[tuple[RegisterPointerSpec, Any, Any]] = []
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
            bbox={"facecolor": (0.0, 0.0, 0.0, 0.6), "edgecolor": "none", "pad": 1.5},
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
        bbox={"facecolor": (0.0, 0.0, 0.0, 0.65), "edgecolor": "none", "pad": 1.8},
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
    im_vga = ax_vga.imshow(
        render_vga_text(vga0, font_vga),
        interpolation="nearest",
        origin="upper",
    )

    if overlay_mappings:
        mask_flat, mapping_visible = build_mapping_mask(overlay_mappings, pixels)
        if mapping_visible:
            mapping_mask_full = mask_flat.reshape(args.height, args.width)

            try:
                data_flat, data_mask_flat = build_mapping_overlay_data(
                    qmp,
                    mapping_visible,
                    pixels,
                )
            except Exception:
                data_flat = data_mask_flat = None
            if (
                data_flat is not None
                and data_mask_flat is not None
                and mask_has_values(data_mask_flat)
            ):
                mapping_data_full = data_flat.reshape(args.height, args.width)
                mapping_data_mask_full = data_mask_flat.reshape(args.height, args.width)
                im.set_data(view_slice_with_overlay())
                im_mag.set_data(
                    render_text_panel(magnifier_block_at(sx0, sy0), font_small)
                )

            color_entries = [(0.0, 0.0, 0.0, 0.0)]
            for mapping in mapping_visible:
                color = region_color_lookup.get(
                    (mapping.start, mapping.end, mapping.label),
                    "#cccccc",
                )
                rgba = mcolors.to_rgba(color, alpha=0.25)
                color_entries.append(rgba)
            overlay_cmap = mcolors.ListedColormap(color_entries, name="memory_layout")
            overlay_norm = mcolors.BoundaryNorm(
                np.arange(len(color_entries) + 1) - 0.5,
                len(color_entries),
            )
            mapping_overlay_im = ax.imshow(
                mapping_mask_full[vy0:vy0 + vH, vx0:vx0 + vW],
                cmap=overlay_cmap,
                norm=overlay_norm,
                interpolation="nearest",
                origin="upper",
                zorder=im.get_zorder() + 0.1,
            )
            legend_handles: list[Any] = []
            for idx, mapping in enumerate(mapping_visible, start=1):
                rgba = overlay_cmap(idx)
                handle = patches.Patch(
                    facecolor=rgba,
                    edgecolor=(rgba[0], rgba[1], rgba[2], min(1.0, rgba[3] + 0.2)),
                    label=(
                        f"{mapping.label} "
                        f"(0x{mapping.start:08X}-0x{mapping.end:08X})"
                    ),
                )
                legend_handles.append(handle)
            if legend_handles:
                ax_legend.cla()
                ax_legend.set_axis_off()
                ax_legend.legend(
                    handles=legend_handles,
                    loc="upper left",
                    fontsize=8,
                    framealpha=0.65,
                    title="Memory layout",
                )

    refresh_axes()

    register_state: dict[str, int] = {}

    def update_pointer_plot(registers: dict[str, int]) -> None:
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

            label_x = float(clamp(round(lx + 2), 0, max(0, vW - 1)))
            label_y = float(clamp(round(ly + 2), 0, max(0, vH - 1)))
            label_text.set_position((label_x, label_y))
            label_text.set_visible(True)

    try:
        register_state = parse_register_dump(qmp.hmp("info registers"))
    except Exception as exc:  # pragma: no cover - log noise in tests
        logger.debug("Failed to fetch initial register dump: %s", exc)
    update_pointer_plot(register_state)

    def on_key(event: KeyEvent) -> None:
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

    def on_move(event: MouseEvent) -> None:
        nonlocal sel_cx, sel_cy
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        sel_cx = vx0 + round(event.xdata)
        sel_cy = vy0 + round(event.ydata)

    def on_scroll(event: MouseEvent) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        cx = vx0 + round(event.xdata)
        cy = vy0 + round(event.ydata)
        if event.button == "up":
            set_view((cx, cy), 0.8)
        elif event.button == "down":
            set_view((cx, cy), 1.25)

    cid_k = fig.canvas.mpl_connect("key_press_event", on_key)
    cid_s = fig.canvas.mpl_connect("scroll_event", on_scroll)
    cid_m = fig.canvas.mpl_connect("motion_notify_event", on_move)

    def update(_: Event) -> tuple[object, ...]:
        nonlocal need_axes_refresh, register_state
        im.set_data(view_slice_with_overlay())
        if mapping_overlay_im is not None and mapping_mask_full is not None:
            mapping_overlay_im.set_data(mapping_mask_full[vy0:vy0 + vH, vx0:vx0 + vW])
            mapping_overlay_im.set_extent((-0.5, vW - 0.5, vH - 0.5, -0.5))
        if need_axes_refresh:
            refresh_axes()

        cx = clamp(sel_cx, 0, args.width - 1)
        cy = clamp(sel_cy, 0, args.height - 1)
        sx = clamp(cx - PANEL_SIZE // 2, 0, args.width - PANEL_SIZE)
        sy = clamp(cy - PANEL_SIZE // 2, 0, args.height - PANEL_SIZE)

        rect.set_xy((sx - vx0, sy - vy0))
        rect.set_width(PANEL_SIZE)
        rect.set_height(PANEL_SIZE)

        im_mag.set_data(render_text_panel(magnifier_block_at(sx, sy), font_small))

        addr = cy * args.width + cx
        region = find_region_for_address(addr)
        if region is None:
            hover_lines = [
                "Unlabeled memory",
                f"0x{addr:08X} ({format_kib_from_bytes(addr)})",
            ]
        else:
            hover_lines = [
                region.label,
                f"0x{addr:08X} ({format_kib_from_bytes(addr)})",
            ]
        hover_text.set_text("\n".join(hover_lines))

        try:
            vga = qmp_read_b800(qmp)
            im_vga.set_data(render_vga_text(vga, font_vga))
        except Exception as exc:  # pragma: no cover - live viewer only
            logger.debug("Failed to refresh VGA buffer: %s", exc)

        try:
            reg_txt = qmp.hmp("info registers")
        except Exception as exc:  # pragma: no cover - live viewer only
            logger.debug("Failed to refresh register dump: %s", exc)
        else:
            register_state = parse_register_dump(reg_txt)

        update_pointer_plot(register_state)

        artists: list[object] = [im]
        if mapping_overlay_im is not None:
            artists.append(mapping_overlay_im)
        artists.extend([im_mag, im_vga, rect, hover_text])
        for _, marker_line, label_text in pointer_artists:
            artists.extend((marker_line, label_text))
        return tuple(artists)

    interval_ms = int(1000 / max(1, args.fps))
    anim = FuncAnimation(
        fig,
        update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    _ = (cid_k, cid_s, cid_m, anim)


if __name__ == "__main__":
    main()
