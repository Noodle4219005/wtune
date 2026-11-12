"""Resolution of every figure `wtune post` writes: each savefig passes its own base dpi through
dpi(), which scales it by WTUNE_POST_DPI_SCALE (default 2). Layout and font sizes are in inches
and points, so the scale only sharpens the image."""
import os

DEFAULT_SCALE = 2.0


def dpi(base: float = 100) -> int:
    try:
        scale = float(os.environ.get("WTUNE_POST_DPI_SCALE", "") or DEFAULT_SCALE)
    except ValueError:
        scale = DEFAULT_SCALE
    return max(1, int(round(base * max(scale, 0.1))))
