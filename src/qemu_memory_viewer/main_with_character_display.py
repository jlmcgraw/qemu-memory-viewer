"""Compatibility entry point exposing the character display viewer.

The legacy ``main_with_character_display`` module bundled a bespoke
Matplotlib event loop that duplicated most of the functionality now
implemented in :mod:`qemu_memory_viewer.main`.  Downstream users still import
that module name though, so we retain it as a thin, well-documented wrapper
that forwards to the new implementation while providing a couple of helper
utilities that callers historically relied upon.
"""

from __future__ import annotations

from . import main as _core

__all__ = [
    "PANEL_SIZE",
    "VGA_COLS",
    "VGA_ROWS",
    "bytes_to_cp437_lines",
    "clamp",
    "compute_marker_rows",
    "main",
    "pick_mono_font",
    "render_text_panel",
]

PANEL_SIZE = _core.PANEL_SIZE
"""Alias for the CP437 panel width and height used by the viewer."""

VGA_ROWS = _core.VGA_ROWS
"""Number of rows in the VGA text buffer."""

VGA_COLS = _core.VGA_COLS
"""Number of columns in the VGA text buffer."""

# Re-export the core helpers so third-party tooling keeps working.  They carry
# their own docstrings and the heavy lifting continues to live in ``main``.
clamp = _core.clamp
pick_mono_font = _core.pick_mono_font
bytes_to_cp437_lines = _core.bytes_to_cp437_lines
render_text_panel = _core.render_text_panel


def compute_marker_rows(
    markers_kib: list[int], full_width: int, full_height: int
) -> list[int]:
    """Convert legacy KiB marker positions into row indices.

    The original character display viewer annotated the Y axis with labels that
    represented well-known offsets into guest memory (for example, ``640 KiB``
    and ``768 KiB``).  Each label was specified in KiB and needed to be mapped
    onto a row within the 2-D byte grid that is ``full_width`` bytes wide and
    ``full_height`` rows tall.  The conversion is a straightforward byte offset
    calculation followed by a clamp so that markers outside the drawable region
    simply snap to the closest valid row instead of raising an exception.
    """
    if full_width <= 0:
        msg = "The viewer width must be a positive integer"
        raise ValueError(msg)
    if full_height <= 0:
        msg = "The viewer height must be a positive integer"
        raise ValueError(msg)

    max_index = full_width * full_height - 1
    rows: list[int] = []
    for marker in markers_kib:
        byte_offset = marker * 1024
        bounded = _core.clamp(byte_offset, 0, max_index)
        rows.append(bounded // full_width)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Run the interactive viewer with the CP437 character overlay.

    Parameters
    ----------
    argv:
        Present for backwards compatibility; ignored because the modern
        implementation in :mod:`qemu_memory_viewer.main` performs its own
        :mod:`argparse` processing directly from :data:`sys.argv`.
    """
    # ``argv`` is accepted solely to retain the historical signature.  The
    # core ``main`` function continues to own CLI parsing.
    del argv
    _core.main()
