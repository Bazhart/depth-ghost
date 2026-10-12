#!/usr/bin/env python3
"""
video_metrics.py — how steady is a depth-ghost video? Compares one or more
renders against their source footage. Cut transitions are excluded (detected on
the source the same way depth_ghost_smooth.py does).

  lighting pump  mean / worst frame-to-frame change of overall brightness (/255)
  jumps>1        frame-to-frame brightness changes above 1/255 (visible pumps)
  shimmer        mean change on pixels that are still in the footage
  jitter         mean per-pixel acceleration (lower = more fluid)

usage: python video_metrics.py SOURCE RENDER [RENDER ...] [--start S] [--per-shot]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_ghost_smooth import detect_cuts, open_at  # noqa: E402


def load(path, first, n, size=(1029, 540)):
    cap = open_at(path, first)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float16))
    return np.array(out, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("renders", nargs="+")
    p.add_argument("--start", type=float, default=0, help="second of the source where the renders begin")
    p.add_argument("--per-shot", action="store_true")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.renders[0])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    first = int(round(a.start * fps))
    cuts = [c - first for c in detect_cuts(a.source, first, first + n)]
    src = load(a.source, first, n)
    n = len(src)
    keep = np.ones(n - 1, bool)          # transitions t -> t+1
    for c in cuts:
        keep[c - 1] = False
    static = np.abs(np.diff(src, axis=0)) < 2
    bounds = [0] + cuts + [n]
    print(f"{n} frames, {len(bounds) - 1} shot(s)")
    print(f"{'':34s} {'pump':>6s} {'worst':>6s} {'jumps>1':>8s} {'shimmer':>8s} {'jitter':>7s}")
    for r in a.renders:
        v = load(r, 0, n)
        m = v.mean(axis=(1, 2))
        pump = np.abs(np.diff(m))
        d1 = np.abs(np.diff(v, axis=0))
        sh = np.array([d1[t][static[t]].mean() if static[t].any() else 0 for t in range(n - 1)])
        acc = np.abs(v[2:] - 2 * v[1:-1] + v[:-2]).mean(axis=(1, 2))
        akeep = keep[1:] & keep[:-1]
        print(f"{Path(r).name[:34]:34s} {pump[keep].mean():6.2f} {pump[keep].max():6.2f} {(pump[keep] > 1).sum():8d} "
              f"{sh[keep].mean():8.2f} {acc[akeep].mean():7.2f}")
        if a.per_shot:
            for s, e in zip(bounds[:-1], bounds[1:]):
                seg = slice(s, max(s, e - 1))
                if e - s < 3:
                    continue
                print(f"   shot {s / fps:5.1f}-{e / fps:5.1f}s {'':12s} {pump[seg].mean():6.2f} {pump[seg].max():6.2f} "
                      f"{(pump[seg] > 1).sum():8d} {sh[seg].mean():8.2f}")


if __name__ == "__main__":
    main()
