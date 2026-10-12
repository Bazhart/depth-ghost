#!/usr/bin/env python3
"""
depth_ghost_smooth.py — the ghost-depth look for VIDEO, with shot-aware temporal
smoothing. depth_ghost.py does the model and the grade; this adds:

  0. Shots: every hard cut starts a new, independent clip. Nothing is smoothed,
     blended or normalised across a cut — each shot gets its own treatment.
  Per shot:
  1. Global range stabilisation: the model re-guesses its depth scale every
     frame, so the whole frame pumps. Each frame's robust range is mapped onto a
     temporally smoothed curve of itself (anchored, so no drift).
  2. Motion-compensated consensus filter: optical flow from the footage warps the
     depth of the +-K neighbour frames onto each frame; votes from the same
     surface are combined by consensus (agreement with the per-pixel median), so
     one-frame model hiccups are outvoted. Only the broad part of the correction
     is kept (guided filter): silhouettes stay crisp and --detail can't turn warp
     mismatch into specks. Where fast motion leaves too few votes it falls back
     to the frame's own depth.
  3. Still-area blend: zero-phase temporal blend wherever depth barely changes;
     moving things and silhouettes get no blend (no smear, no lag).
  4. One fixed black/white range for the whole shot (no grade pumping).
  5. Highlight shoulder (--knee/--tail/--bend): the nearest objects (mics,
     railings) keep grey levels instead of blowing out to white.
  6. Regional deflicker: brightness corrections on a coarse grid, measured only
     on pixels that are still in the footage (real motion can't fool it),
     smoothed in time, capped.

Usage
  python depth_ghost_smooth.py clip.mp4 -o clip_depth.mp4 --model base --res 770 \\
      --gamma 1.4 --black 0.07 --exposure 1.4 --detail 2.5 --smooth 0 --temporal 0.95 \\
      --knee 0.70 --tail 0.06 --hi-pct 99.2
Depth is cached per frame (default ~/.depth-ghost/cache/<name>_<model>_<res>), and
the smoothed depth too, so re-grading with different look flags takes minutes.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_ghost as dg  # noqa: E402


# ---------------------------------------------------------------- helpers

def gsmooth1d(x, sigma, mode="edge"):
    """Gaussian smoothing of a 1-D series (edge/reflect padded, same length)."""
    x = np.asarray(x, np.float64)
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    r = int(3 * sigma + 0.5)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    if mode == "reflect" and len(x) <= r:
        mode = "edge"
    return np.convolve(np.pad(x, r, mode=mode), k, mode="valid")[: len(x)]


def guided(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = mean(I), mean(p)
    a = (mean(I * p) - mI * mp) / (mean(I * I) - mI * mI + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def make_tone_curve(knee, peak, tail, bend):
    """dg.tone_curve with an adjustable highlight shoulder.

    knee: brightness where the roll-off starts (depth_ghost's fixed value is 0.8).
    tail > 0: bend over --bend of input past the knee, then keep slope `tail`, so
    the nearest objects stay dim but keep internal gradation (mic body vs pop
    filter). tail = 0: roll off toward `peak` (classic shoulder).
    """
    def tone_curve(args, samples=2048):
        n = np.linspace(0, 1, samples)
        d = np.clip((n - args.black) / max(1 - args.black, 1e-6), 0, 1)
        d = d ** args.gamma
        d = d * args.exposure
        if tail > 0:
            x = np.maximum(d - knee, 0)
            d = np.where(d < knee, d, knee + (1 - tail) * bend * (1 - np.exp(-x / bend)) + tail * x)
        else:
            d = np.where(d < knee, d, knee + (peak - knee) * (1 - np.exp(-(d - knee) / (peak - knee))))
        if args.lift > 0:  # same background lift as depth_ghost.tone_curve
            band = args.lift_band
            far = np.log1p(20 * np.clip(n / band, 0, 1)) / np.log1p(20) * args.lift
            p = int(np.argmax(n >= band))
            catch = np.nonzero((d >= far[p] + 0.25 * (1 - far[p])) & (n > n[p]))[0]
            q = catch[0] if len(catch) else samples - 1
            bridge = np.linspace(far[p], d[q], q - p + 1)
            d = np.maximum(d, np.concatenate([far[:p], bridge, np.zeros(samples - q - 1)]))
        return n, np.maximum.accumulate(d)
    return tone_curve


def open_at(video, first):
    cap = cv2.VideoCapture(str(video))  # applies iPhone rotation metadata
    for _ in range(first):  # grab sequentially: frame-accurate, unlike seeking
        cap.grab()
    return cap


# ---------------------------------------------------------------- shots

def detect_cuts(video, first, last, diff=10.0, spike=3.0, hist=0.10):
    """Hard cuts = a one-frame spike in pixel change (not a sustained whip pan)
    that also changes the colour histogram. Returns absolute frame indices
    where a new shot starts."""
    cap = open_at(video, first)
    g, h = [], []
    for _ in range(last - first):
        ok, f = cap.read()
        if not ok:
            break
        s = cv2.resize(f, (160, 84), interpolation=cv2.INTER_AREA)
        g.append(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32))
        hh = cv2.calcHist([s], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
        h.append(hh / max(hh.sum(), 1))
    cap.release()
    n = len(g)
    d, hd = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        d[t] = np.abs(g[t] - g[t - 1]).mean()
        hd[t] = 0.5 * np.abs(h[t] - h[t - 1]).sum()
    cuts = []
    for t in range(1, n):
        nb = np.r_[d[max(1, t - 6):t], d[t + 1:t + 7]]
        base = np.median(nb) if nb.size else 0.0
        if d[t] > diff and d[t] / (base + 1) > spike and hd[t] > hist:
            cuts.append(first + t)
    return cuts


# ---------------------------------------------------------------- per-shot stages

def stabilize(depth, grey, a):
    """Stages 1+2. depth: list of float16 HxW, grey: list of uint8 HxW."""
    n = len(depth)
    H, W = depth[0].shape
    # 1. global range stabilisation
    lo = np.array([np.percentile(d.astype(np.float32), 5) for d in depth])
    hi = np.array([np.percentile(d.astype(np.float32), 97) for d in depth])
    ls, hs = gsmooth1d(lo, a.global_sigma), gsmooth1d(hi, a.global_sigma)
    scale = (hs - ls) / np.maximum(hi - lo, 1e-6)
    depth = [((d.astype(np.float32) - lo[i]) * scale[i] + ls[i]).astype(np.float16) for i, d in enumerate(depth)]
    if a.stab_k <= 0 or n < 2:
        return depth, float(np.median(np.abs(scale - 1)))

    sub = np.stack([d[::4, ::4] for d in depth[::max(1, n // 40)]]).astype(np.float32)
    glo, ghi = np.percentile(sub, 1), np.percentile(sub, 99.5)
    R = max(ghi - glo, 1e-6)

    # 2. motion-compensated consensus filter
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    small = [cv2.resize(g, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for g in grey]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ts = max(1.8, 0.6 * a.stab_k)
    tw = {j: float(np.exp(-j * j / (2 * ts ** 2))) for j in range(1, a.stab_k + 1)}
    out_all = []
    for i in range(n):
        d_i = depth[i].astype(np.float32)
        g_i = grey[i].astype(np.float32)
        stack, base_w = [d_i], [np.ones((H, W), np.float32)]
        for j in range(-a.stab_k, a.stab_k + 1):
            m = i + j
            if j == 0 or not 0 <= m < n:
                continue
            flow = dis.calc(small[i], small[m], None)          # i(y,x) ~ m(y+fy, x+fx)
            flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) * 2
            mx, my = gx + flow[..., 0], gy + flow[..., 1]
            wd = cv2.remap(depth[m].astype(np.float32), mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            wg = cv2.remap(grey[m], mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
            photo = np.exp(-((np.abs(wg - g_i) / 255 / a.photo_sigma) ** 2))
            wd[photo < 0.5] = np.nan          # only votes that land on the same surface
            stack.append(wd)
            base_w.append(tw[abs(j)] * photo)
        stack, base_w = np.stack(stack), np.stack(base_w)
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
        agree = np.exp(-((np.abs(stack - med) / R / a.depth_sigma) ** 2))
        w = np.where(np.isnan(stack), 0, base_w * agree)
        wsum = w.sum(0)
        out = (w * np.nan_to_num(stack)).sum(0) / np.maximum(wsum, 1e-6)
        out = np.where(wsum > 1e-3, out, d_i)
        # fast motion: too few votes -> fade back to the frame's own depth
        votes = (~np.isnan(stack)).sum(0)
        nv = 2 * a.stab_k + 1
        trust = cv2.GaussianBlur(np.clip((votes - 0.3 * nv) / (0.4 * nv), 0, 1).astype(np.float32), (0, 0), 3)
        # keep only the broad part of the correction (no specks, crisp silhouettes)
        guide = np.clip((d_i - glo) / R, 0, 1).astype(np.float32)
        corr = guided(guide, (trust * (out - d_i)).astype(np.float32), a.corr_radius, 1e-3)
        out_all.append((d_i + corr).astype(np.float16))
    return out_all, float(np.median(np.abs(scale - 1)))


def still_blend(depth, a):
    """Stage 3: zero-phase blend where depth barely changes frame to frame."""
    n = len(depth)
    if a.still_cap <= 0 or n < 2:
        return depth
    sub = np.stack([d[::4, ::4] for d in depth[::max(1, n // 40)]]).astype(np.float32)
    R = max(np.percentile(sub, 99.5) - np.percentile(sub, 1), 1e-6)

    def step(prev, d):
        w = a.still_cap * np.exp(-((np.abs(d - prev) / R / a.still_tol) ** 2))
        return w * prev + (1 - w) * d

    F, prev = [None] * n, None
    for t in range(n):
        d = depth[t].astype(np.float32)
        prev = d if prev is None else step(prev, d)
        F[t] = prev.astype(np.float16)
    out, prev = [None] * n, None
    for t in range(n - 1, -1, -1):
        d = depth[t].astype(np.float32)
        prev = d if prev is None else step(prev, d)
        out[t] = (0.5 * (F[t].astype(np.float32) + prev)).astype(np.float16)
        F[t] = None
    return out


def grade(depth, a):
    """Stages 4+5: fixed range for the shot, depth_ghost grade + its still-pixel blend."""
    n = len(depth)
    if a.fixed_range:
        idx = range(0, n, max(1, n // 40))
        p = np.array([np.percentile(depth[i].astype(np.float32)[::2, ::2], [a.lo_pct, a.hi_pct]) for i in idx])
        lo, hi = np.median(p[:, 0]), np.median(p[:, 1])
    out, prev = [], None
    plo = phi = None
    for t in range(n):
        d = depth[t].astype(np.float32)
        if not a.fixed_range:  # eased range, like depth_ghost.py
            flo, fhi = dg.percentile_range(d, a.lo_pct, a.hi_pct)
            plo, phi = (flo, fhi) if plo is None else (a.temporal * plo + (1 - a.temporal) * flo,
                                                       a.temporal * phi + (1 - a.temporal) * fhi)
            lo, hi = plo, phi
        if prev is not None and a.temporal > 0:
            diff = np.abs(d - prev) / max(hi - lo, 1e-6)
            w = 0.6 * a.temporal * np.exp(-((diff / 0.04) ** 2))
            d = w * prev + (1 - w) * d
        prev = d
        out.append(dg.ghost_grade(d, None, lo, hi, a))
    return out, (lo, hi)


def deflicker(graded, grey, a):
    """Stage 6: regional deflicker measured on still pixels only."""
    n = len(graded)
    if a.deflicker_max <= 0 or n < 3:
        return graded, 0.0
    gc, gr = (int(v) for v in a.deflicker_grid.lower().split("x"))
    H, W = graded[0].shape
    hs, ws = H // 4, W // 4
    sm = [cv2.resize(g, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32) for g in graded]
    sg = [cv2.resize(g, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32) for g in grey]
    ys = np.linspace(0, hs, gr + 1).astype(int)
    xs = np.linspace(0, ws, gc + 1).astype(int)
    logr = np.zeros((n, gr, gc))
    for t in range(1, n):
        still = (np.abs(sg[t] - sg[t - 1]) < 3) & (sm[t] > 8) & (sm[t - 1] > 8)
        for r in range(gr):
            for c in range(gc):
                cell = np.s_[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                m = still[cell]
                if m.mean() < 0.05:
                    continue  # nothing still in this cell: no information
                logr[t, r, c] = np.log(sm[t][cell][m].mean() / sm[t - 1][cell][m].mean())
    L = np.cumsum(logr, axis=0)
    smoothL = np.empty_like(L)
    for r in range(gr):
        for c in range(gc):
            smoothL[:, r, c] = gsmooth1d(L[:, r, c], a.deflicker_sigma, mode="reflect")
    gain = np.clip(np.exp(smoothL - L), 1 - a.deflicker_max, 1 + a.deflicker_max)
    out = []
    for t in range(n):
        field = cv2.resize(gain[t].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        if gc * gr > 1:
            field = cv2.GaussianBlur(field, (0, 0), min(W / gc, H / gr) / 3)  # no visible cell edges
        out.append(np.clip(graded[t].astype(np.float32) * field, 0, 255).astype(np.uint8))
    return out, float(np.median(np.abs(gain - 1)))


# ---------------------------------------------------------------- driver

SMOOTH_KEYS = ("stab_k", "global_sigma", "depth_sigma", "photo_sigma", "corr_radius", "still_cap", "still_tol")


def read_shot(cap, s, e, a, cache, box):
    """Read frames s..e-1 (cap is positioned at s): cached depth (computed if
    missing) + grey footage at depth resolution."""
    depth, grey, computed = [], [], 0
    for i in range(s, e):
        ok, bgr = cap.read()
        if not ok:
            break
        if a.height and a.height != bgr.shape[0]:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (round(w * a.height / h) // 2 * 2, a.height),
                             interpolation=cv2.INTER_AREA if a.height < h else cv2.INTER_LANCZOS4)
        f = cache / f"{i:06d}.npy"
        if f.exists():
            d = np.load(f).astype(np.float16)
        else:
            if box.get("model") is None:
                box["model"] = dg.DepthModel(a.model, a.res, a.device)
            d = box["model"](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.float16)
            np.save(f, d)
            computed += 1
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if g.shape != d.shape:
            g = cv2.resize(g, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
        depth.append(d)
        grey.append(g)
    return depth, grey, computed


def main():
    p = argparse.ArgumentParser(description="Ghost depth video with shot-aware temporal smoothing.")
    p.add_argument("input", help="video file")
    p.add_argument("-o", "--out", help="output .mp4 (default <name>_depth_smooth.mp4 next to input)")
    g = p.add_argument_group("look (same meaning as depth_ghost.py)")
    g.add_argument("--model", choices=dg.MODELS, default="small")
    g.add_argument("--res", type=int, default=518)
    g.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    g.add_argument("--gamma", type=float, default=1.8)
    g.add_argument("--black", type=float, default=0.08)
    g.add_argument("--exposure", type=float, default=1.05)
    g.add_argument("--lo-pct", type=float, default=1)
    g.add_argument("--hi-pct", type=float, default=99.5)
    g.add_argument("--smooth", type=int, default=8, help="spatial edge-aware smoothing radius, 0 = off")
    g.add_argument("--detail", type=float, default=0)
    g.add_argument("--relief", type=float, default=0)
    g.add_argument("--lift", type=float, default=0)
    g.add_argument("--lift-band", type=float, default=0.35)
    g.add_argument("--grain", type=float, default=0)
    g.add_argument("--height", type=int)
    g.add_argument("--pillarbox", action="store_true")
    g.add_argument("--temporal", type=float, default=0.85, help="still-pixel blend inside the grade (0-1)")
    g.add_argument("--start", type=float, default=0, help="begin at second S")
    g.add_argument("--seconds", type=float, help="only this many seconds")
    g.add_argument("--depth-cache", help="per-frame depth cache dir (default ~/.depth-ghost/cache/<name>_<model>_<res>)")
    h = p.add_argument_group("highlight shoulder")
    h.add_argument("--knee", type=float, default=0.8, help="where the highlight roll-off starts (0.8 = depth_ghost default)")
    h.add_argument("--peak", type=float, default=1.0, help="brightness the roll-off approaches (tail = 0)")
    h.add_argument("--tail", type=float, default=0.0, help=">0: dim nearest objects but keep their gradation, e.g. 0.06")
    h.add_argument("--bend", type=float, default=0.07, help="how quickly the curve bends past the knee (with --tail)")
    s = p.add_argument_group("smoothing (per shot)")
    s.add_argument("--stab-k", type=int, default=3, help="motion-compensated neighbours each side (0 = off). >3 smears fast textureless motion")
    s.add_argument("--global-sigma", type=float, default=10, help="frames; smoothing of the overall depth scale")
    s.add_argument("--depth-sigma", type=float, default=0.08)
    s.add_argument("--photo-sigma", type=float, default=0.05)
    s.add_argument("--corr-radius", type=int, default=12)
    s.add_argument("--still-cap", type=float, default=0.8, help="still-area blend strength (0 = off). >0.85 fades slow distant movers")
    s.add_argument("--still-tol", type=float, default=0.03)
    s.add_argument("--no-fixed-range", dest="fixed_range", action="store_false", help="ease the range per frame instead")
    s.add_argument("--deflicker-grid", default="6x4", help="cols x rows; 1x1 = whole frame (adds shimmer when things move)")
    s.add_argument("--deflicker-max", type=float, default=0.06, help="largest correction, 0 = off")
    s.add_argument("--deflicker-sigma", type=float, default=8)
    c = p.add_argument_group("shots")
    c.add_argument("--cuts", default="auto", help="'auto', 'none', or frame numbers '305,319,...' where new shots start")
    c.add_argument("--cut-diff", type=float, default=10)
    c.add_argument("--cut-spike", type=float, default=3)
    c.add_argument("--cut-hist", type=float, default=0.10)
    a = p.parse_args()

    src = Path(a.input).expanduser()
    dst = Path(a.out).expanduser() if a.out else src.with_name(f"{src.stem}_depth_smooth.mp4")
    cache = Path(a.depth_cache).expanduser() if a.depth_cache else \
        Path.home() / ".depth-ghost" / "cache" / f"{src.stem}_{a.model}_{a.res}{f'_h{a.height}' if a.height else ''}"
    cache.mkdir(parents=True, exist_ok=True)
    dg.tone_curve = make_tone_curve(a.knee, a.peak, a.tail, a.bend)

    probe = cv2.VideoCapture(str(src))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    first = int(round(a.start * fps))
    last = total if not a.seconds else min(total, first + int(round(a.seconds * fps)))

    if a.cuts == "auto":
        cuts = detect_cuts(src, first, last, a.cut_diff, a.cut_spike, a.cut_hist)
    elif a.cuts == "none":
        cuts = []
    else:
        cuts = sorted(int(x) for x in a.cuts.split(",") if first < int(x) < last)
    bounds = [first] + cuts + [last]
    shots = list(zip(bounds[:-1], bounds[1:]))
    print(f"{len(shots)} shot(s): " + ", ".join(f"{s_ / fps:.2f}-{e_ / fps:.2f}s" for s_, e_ in shots), file=sys.stderr)

    scache = None  # smoothed-depth cache, only worth keeping when smoothing is on
    if a.stab_k > 0 or a.still_cap > 0 or a.global_sigma > 0:
        sig = json.dumps({k: getattr(a, k) for k in SMOOTH_KEYS} | {"cache": str(cache), "shots": shots}, sort_keys=True)
        scache = Path(f"{cache}_smoothed_{hashlib.sha1(sig.encode()).hexdigest()[:8]}")
        scache.mkdir(parents=True, exist_ok=True)
        (scache / "params.json").write_text(sig)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg is required for video (brew install ffmpeg)")
    cap = open_at(src, first)
    box, enc, t0, n_out = {}, None, time.time(), 0
    for si, (s_, e_) in enumerate(shots):
        ts = time.time()
        depth, grey, computed = read_shot(cap, s_, e_, a, cache, box)
        if not depth:
            break
        e_ = s_ + len(depth)
        sfiles = [scache / f"{i:06d}.npy" for i in range(s_, e_)] if scache else []
        if sfiles and all(f.exists() for f in sfiles):
            sm, note = [np.load(f) for f in sfiles], "smoothed depth reused"
        else:
            sm, gfix = stabilize(depth, grey, a)
            sm = still_blend(sm, a)
            for f, d in zip(sfiles, sm):
                np.save(f, d)
            note = f"scale steadied {gfix * 100:.1f}%/frame"
        del depth
        graded, (lo, hi) = grade(sm, a)
        del sm
        graded, gmed = deflicker(graded, grey, a)
        del grey
        for gr_ in graded:
            frame = dg.finish(gr_, a)
            frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if enc is None:
                fh, fw = frame.shape[:2]
                fw2, fh2 = fw // 2 * 2, fh // 2 * 2
                cmd = [ffmpeg, "-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw2}x{fh2}", "-r", str(fps), "-i", "-",
                       "-ss", f"{first / fps:.6f}", "-t", f"{(last - first) / fps:.6f}", "-i", str(src),
                       "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
                       "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(dst)]
                enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            enc.stdin.write(np.ascontiguousarray(frame[:fh2, :fw2]).tobytes())
            n_out += 1
        del graded
        print(f"shot {si + 1}/{len(shots)}  frames {s_}-{e_ - 1} ({(e_ - s_) / fps:.1f}s)  "
              f"{'computed ' + str(computed) + ' depth, ' if computed else ''}{note}, "
              f"deflicker {gmed * 100:.1f}%  [{time.time() - ts:.0f}s, total {time.time() - t0:.0f}s]",
              file=sys.stderr, flush=True)
    cap.release()
    if enc is None:
        sys.exit(f"could not read any frames from {src}")
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed while encoding the video")
    print(f"{n_out} frames -> {dst}  (depth cache {cache}, smoothed {scache})", file=sys.stderr)
    print(dst)


if __name__ == "__main__":
    main()
