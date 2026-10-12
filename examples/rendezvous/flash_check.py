"""Photosensitive flash check (simplified WCAG 2.3.1 general-flash rule).

A flash = a pair of opposing changes in relative luminance of >= 10% where the darker
state is below 0.8. More than 3 flashes inside any 1-second window fails. Checked on the
whole frame and on a 4x4 grid (each cell is about the rule's 25%-of-view area).

usage: python flash_check.py VIDEO [VIDEO ...]
"""
import sys

import cv2
import numpy as np


def rel_lum(bgr):
    c = bgr.astype(np.float32) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]


def series(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        L = rel_lum(cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA))
        cells = L.reshape(4, 36, 4, 64).mean(axis=(1, 3)).ravel()
        out.append(np.r_[L.mean(), cells])
    return np.array(out), fps


def max_flashes_per_second(x, fps):
    """Count luminance transitions >= 0.1 (darker side < 0.8), tracking peaks and troughs;
    two opposing transitions make one flash. Returns the worst count in any 1 s window."""
    events, direction, ext = [], 0, x[0]
    for i in range(1, len(x)):
        v = x[i]
        if direction == 0:
            if abs(v - ext) >= 0.1 and min(v, ext) < 0.8:
                direction, ext = (1 if v > ext else -1), v
                events.append(i)
        elif direction > 0:
            if v > ext:
                ext = v
            elif ext - v >= 0.1 and v < 0.8:
                direction, ext = -1, v
                events.append(i)
        else:
            if v < ext:
                ext = v
            elif v - ext >= 0.1 and ext < 0.8:
                direction, ext = 1, v
                events.append(i)
    ev = np.array(events)
    return max((int(((ev >= i) & (ev < i + fps)).sum()) // 2 for i in ev), default=0)


for path in sys.argv[1:]:
    s, fps = series(path)
    whole = max_flashes_per_second(s[:, 0], fps)
    cells = max(max_flashes_per_second(s[:, k], fps) for k in range(1, s.shape[1]))
    verdict = "PASS" if max(whole, cells) <= 3 else "FAIL"
    print(f"{path.split('/')[-1]:40s} max flashes/s  whole frame {whole}  any region {cells}  -> {verdict}")
