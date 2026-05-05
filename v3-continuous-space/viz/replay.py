"""GIF export from a list of pre-rendered RGB frames."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def save_gif(frames: Iterable[np.ndarray], path: str | Path, *, fps: int = 8) -> None:
    """Save a sequence of (H, W, 3) uint8 RGB ndarrays as an animated GIF."""
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("save_gif called with no frames")
    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(round(1000 / max(1, fps)))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
