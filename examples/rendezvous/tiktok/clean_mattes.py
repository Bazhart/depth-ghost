#!/usr/bin/env python3
"""Vision person mattes (portrait crops) -> clean footage-space mattes in frames/person/, plus a stability report.
Per shot: back into the 2058x1080 frame, temporal median over 5 frames, only the singer's own shape (and any
other full-size person), nothing much farther than him in depth, pulled ~5 px inside the silhouette, feathered."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent; PR = HERE.parent
sys.path.insert(0, str(PR))
import remix_cut as rc
from final_cut import W, H
SRC = Path(sys.argv[1]); OUT = PR / "frames" / "person"; OUT.mkdir(exist_ok=True)
off = {int(k): v for k, v in json.load(open(HERE / "person_crop_offsets.json")).items()}
report = {}
for shot, (a, b) in rc.SHOT.items():
    if shot == "B":
        continue
    raw = {}
    for j in range(a, b):
        m = cv2.imread(str(SRC / f"{j:06d}.png"), cv2.IMREAD_UNCHANGED)
        m = cv2.resize(m[..., 0] if m.ndim == 3 else m, (810, 1080), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        full = np.zeros((H, W), np.float32); full[:, off[j]:off[j] + 810] = m
        raw[j] = cv2.resize(full, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    prev = None
    for j in range(a, b):
        win = [raw[k] for k in range(max(a, j - 2), min(b, j + 3))]
        m = np.median(np.stack(win), axis=0)
        n = cv2.resize(rc.load(j)[2], (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        x = rc.TRACK[j] / 2
        bin_ = (m > 0.5).astype(np.uint8)
        nl, lab, st, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
        keep = np.zeros_like(bin_, bool); main_area = 0
        if nl > 1:
            col = lab[:, int(max(0, x - 25)):int(x + 26)]
            cnt = np.bincount(col[col > 0].ravel(), minlength=nl)
            kmain = int(np.argmax(cnt)) if cnt.max() > 0 else int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
            main_area = st[kmain, cv2.CC_STAT_AREA]
            for k in range(1, nl):
                if k == kmain or st[k, cv2.CC_STAT_AREA] > 0.4 * main_area:
                    keep |= lab == k
            pd = float(np.median(n[lab == kmain]))
            keep &= n > pd - 0.22                            # nothing much farther back than the singer
        soft = m * keep
        soft = cv2.erode((soft > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
        soft = cv2.GaussianBlur(soft, (0, 0), 1.6)
        full = cv2.resize(soft, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(OUT / f"{j:06d}.png"), np.clip(full * 255, 0, 255).astype(np.uint8))
        bm = soft > 0.5
        iou = float((bm & prev).sum() / max(1, (bm | prev).sum())) if prev is not None else 1.0
        report[j] = dict(area=round(float(bm.mean()), 4), iou=round(iou, 3))
        prev = bm
json.dump(report, open(HERE / "person_quality.json", "w"))
for shot, (a, b) in rc.SHOT.items():
    if shot == "B": continue
    ar = np.array([report[j]["area"] for j in range(a, b)]); io = np.array([report[j]["iou"] for j in range(a + 1, b)])
    bad = [j for j in range(a + 1, b) if report[j]["iou"] < 0.85]
    print(f"{shot}: area median {np.median(ar):.3f} (min {ar.min():.3f}, max {ar.max():.3f}), frame-to-frame IoU median {np.median(io):.3f} min {io.min():.3f}, unstable frames {len(bad)} {bad[:12]}")
