#!/usr/bin/env python3
"""
RENDËZVOUS — Scan Cut v2. Final compositor (hybrid of A Pure Scan + C Cinematic Ghost).

v2: blackletter "Rendez-Vous" title placed IN the scene on a depth layer (the artist and
near props pass in front of it, the room sits behind), no end text, no credits, and a
different transition at every cut.

One pass renders three deliverables: 16:9 master (2.39:1 letterbox inside 2058x1080),
9:16 (1080x1920) and 1:1 (1080x1080). Every frame is a pure function of its index:
grain is seeded by frame number, there is no other randomness.

usage: python final_cut.py OUT_DIR [--preview 30,540,...] [--title-font PATH]
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = os.environ.get("DG_SOURCE", str(Path(__file__).resolve().parent / "source.mov"))   # the edit you start from
AUDIO = os.environ.get("DG_AUDIO", str(Path(__file__).resolve().parent / "audio" / "track.wav"))   # the only audio: starts at 0, cut at the end, full level
TITLE_ON = False                               # title removed (v4); code kept for later
GHOST = str(Path(__file__).resolve().parent / "ghost_base_lift15.mp4")   # v8: background lift 0.15 (was 0.30) so far things fall back into black
SCACHE = Path(os.path.expanduser(os.environ.get("DG_DEPTH_CACHE", "~/.depth-ghost/cache/source_smoothed")))   # depth_ghost_smooth.py's smoothed depth
HERE = Path(__file__).resolve().parent
SHOTS = [(0, 305), (305, 319), (319, 538), (538, 715), (715, 919), (919, 1151), (1151, 1318), (1318, 1407)]
N, FPS, W, H = 1407, 30, 2058, 1080
BAR = int(round((H - W / 2.39) / 2))          # 2.39:1 letterbox inside the master
FREEZE = 1376                                  # final freeze frame (shot 8)
MONO = ("/System/Library/Fonts/SFNSMono.ttf", 0)
WARM_LINE = np.array([214, 232, 244], np.float32)   # BGR warm white: scan lines

# One transition per cut, all different. (kind, length in frames into the new shot)
TRANSITIONS = {305: ("far_first", 6), 319: ("whip", 8), 538: ("sweep", 14), 715: ("depth_dissolve", 12),
               919: ("defocus", 10), 1151: ("scanline", 10), 1318: ("punch", 8)}

# Title
TITLE = "Rendez-Vous"
TITLE_DEPTH = 0.28                             # depth plane: anything nearer passes in front of the letters
TITLE_IN, TITLE_OUT = 56, 150                  # after the opening scan; gone before shot 1 ends
ICE = (246, 234, 222)                          # BGR icy fill
NAVY = (82, 52, 30)                            # BGR thin navy outline


def shot_of(i):
    return next(k for k, (s, e) in enumerate(SHOTS) if s <= i < e)


def ease(u):  # smootherstep: eases in and out, never overshoots
    u = float(np.clip(u, 0, 1))
    return u * u * u * (u * (6 * u - 15) + 10)


def tc(i):
    s, f = divmod(i, FPS)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def gauss1d(x, sigma):
    r = int(3 * sigma + 0.5)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2)); k /= k.sum()
    return np.convolve(np.pad(np.asarray(x, float), r, mode="edge"), k, mode="valid")


# ------------------------------------------------------------------ depth, per-shot range
RANGE = {}
for k, (s, e) in enumerate(SHOTS):
    idx = range(s, e, max(1, (e - s) // 15))
    p = np.array([np.percentile(np.load(SCACHE / f"{i:06d}.npy").astype(np.float32)[::4, ::4], [1, 99.2]) for i in idx])
    RANGE[k] = np.median(p, axis=0)


def depth(i):
    lo, hi = RANGE[shot_of(i)]
    return np.clip((np.load(SCACHE / f"{i:06d}.npy").astype(np.float32) - lo) / (hi - lo), 0, 1)


# ------------------------------------------------------------------ audio hits (real transients, no BPM guess)
def audio_hits():
    raw = subprocess.run(["ffmpeg", "-v", "error", "-t", f"{N / FPS:.3f}", "-i", AUDIO, "-vn", "-ac", "1", "-ar", "22050",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    sr, hop, nfft = 22050, 256, 1024
    fr = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop] * np.hanning(nfft)
    m = np.log1p(np.abs(np.fft.rfft(fr, axis=1)) * 10)
    env = np.maximum(np.diff(m, axis=0), 0).sum(1)
    t = (np.arange(len(env)) + 1) * hop / sr
    per = np.zeros(N)
    np.maximum.at(per, np.minimum((t * FPS).astype(int), N - 1), env)
    z = (per - np.median(per)) / (per.std() + 1e-9)
    hits, last = [], -99
    for i in range(1, N - 1):
        lo_, hi_ = max(0, i - 6), min(N, i + 7)
        if z[i] > 2.2 and z[i] == z[lo_:hi_].max() and i - last >= 12:   # >= 0.4 s apart: never above 2.5/s
            hits.append(i); last = i
    return hits


HITS = audio_hits()


def pulse(i):
    s0 = SHOTS[shot_of(i)][0]
    return min(1.0, sum(np.exp(-(i - h) / 5.0) for h in HITS if s0 <= h <= i and i - h < 30))  # resets at cuts


# ------------------------------------------------------------------ subject tracking for the vertical / square crops
def track():
    """Follow the main person, not the nearest object, inside each shot only."""
    xs = np.arange(0, W, 8) + 4.0
    center = np.exp(-((xs - W / 2) / (0.45 * W)) ** 2)
    prof = np.zeros((N, len(xs)))
    for i in range(N):
        d = np.load(SCACHE / f"{min(i, FREEZE):06d}.npy", mmap_mode="r")[::8, ::8].astype(np.float32)
        lo, hi = RANGE[shot_of(i)]
        n = np.clip((d - lo) / (hi - lo), 0, 1)
        w = np.exp(-((n - 0.58) / 0.16) ** 2) * (n > 0.3)
        prof[i] = gauss1d(w.sum(0) * center, 6)
    out = np.zeros(N)
    for s, e in SHOTS:
        x = int(np.argmax(prof[s:e].sum(0)))
        path = []
        for i in range(s, e):
            lo_, hi_ = max(0, x - 40), min(len(xs), x + 41)
            seg = prof[i, lo_:hi_]
            if seg.max() > 0:
                x = int(np.clip(lo_ + int(np.argmax(seg)), x - 4, x + 4))
            path.append(xs[x])
        out[s:e] = gauss1d(path, 12) if e - s > 3 else path
    out[FREEZE:] = out[FREEZE]
    return out


# Singer position (x as % of frame width), marked by eye on keyframes: the person at the mic.
# Shots 1, 5, 7, 8: artist in the varsity jacket; 2, 4: standing figure at the mic on the stairs
# set; 3: the one at the mic stand (left), not the man at the desk; 6: the close-up singer.
SINGER = {
    2: 70, 30: 68, 60: 70, 90: 71, 120: 69, 150: 70, 180: 71, 210: 73, 240: 69, 270: 69, 302: 69,
    307: 64, 316: 63,
    321: 27, 349: 28, 379: 25, 409: 25, 439: 27, 469: 30, 499: 33, 529: 38, 535: 39,
    540: 60, 568: 62, 598: 59, 628: 66, 658: 63, 688: 63, 712: 57,
    717: 80, 730: 81, 745: 81, 760: 76, 775: 66, 790: 68, 805: 79, 820: 74, 835: 63, 850: 64,
    865: 56, 880: 58, 895: 55, 905: 55, 916: 60,
    921: 64, 949: 58, 979: 55, 1009: 55, 1039: 54, 1069: 57, 1099: 54, 1129: 53, 1148: 56,
    1153: 70, 1181: 72, 1211: 71, 1241: 73, 1271: 70, 1301: 71, 1315: 69,
    1320: 62, 1334: 63, 1348: 69, 1362: 70, 1376: 75,
}


def track_singer():
    """Crop centre for the phone formats: interpolate the singer marks, smooth inside each shot only."""
    out = np.zeros(N)
    for s, e in SHOTS:
        ks = sorted(k for k in SINGER if s <= k < e)
        path = np.interp(np.arange(s, e), ks, [SINGER[k] / 100 * W for k in ks])
        out[s:e] = gauss1d(path, 8) if e - s > 3 else path
    out[FREEZE:] = out[FREEZE]
    return out


# ------------------------------------------------------------------ virtual handheld camera (16:9 master)
# Beat grid of the track (140 BPM, half-time feel: one beat = 0.857 s), detected from the audio.
_bt = open(HERE / "beats.txt").read().split("\n")
BEAT_F = np.array([int(round(float(x) * FPS)) for x in _bt[2:] if x.strip()])
DOWN_F = BEAT_F[0::4]                          # bar downbeats (strongest kicks)
CUT_F = [s for s, _ in SHOTS[1:]]

# 808 + open-hat hits, measured on the track (seconds): the groove the camera reacts to.
JOINT_T = [0.84, 1.70, 2.54, 3.41, 3.62, 4.06, 4.27, 4.86, 5.12, 5.54, 5.97, 6.40, 6.83, 7.05, 7.70, 8.55, 9.41,
           10.27, 10.91, 11.12, 11.64, 11.98, 12.41, 13.18, 13.68, 14.25, 14.56, 15.41, 16.25, 17.12, 17.76, 17.98,
           18.62, 18.83, 20.12, 20.33, 20.54, 22.25, 22.88, 23.10, 23.98, 24.61, 24.83, 25.47, 25.69, 26.53, 26.97,
           27.19, 27.41, 28.27, 28.70, 31.23, 31.47, 31.70, 31.89, 32.13, 32.33, 32.55, 35.02, 35.77, 35.98, 37.12,
           37.69, 37.87, 38.02, 38.34, 38.55, 39.19, 39.40, 41.11, 41.98, 42.84, 45.05, 46.69]
# The "dun dun dun": three hard, evenly spaced 808 + open-hat hits. Step zoom, step zoom, crash.
# (The 11.64 s triple sits in shot 3 before the drop, which stays framed as it was.)
TRIPLETS = [(5.12, 5.54, 5.97), (20.12, 20.33, 20.54), (26.97, 27.19, 27.41), (31.23, 31.47, 31.70),
            (31.89, 32.13, 32.33), (37.69, 37.87, 38.02)]
DROP_F = 411                                   # 13.70 s: the 808 slams back in after its break (12.91-13.70 s, measured): zoom-in whip in shot 3
WHIP_F = 1234                                  # 41.15 s: shot 7 crash zoom on the 808's return after its second break (measured)
JOINT_F = sorted({int(round(t * FPS)) for t in JOINT_T})
TRIP_F = [tuple(int(round(t * FPS)) for t in g) for g in TRIPLETS]
TRIP_HITS = {f for g in TRIP_F for f in g}

# Per shot: e = operator energy (handheld amount), comp = where the singer sits across the frame (0-1),
# zoom = slow zoom drift (start, end), tilt = vertical move in px (start, end), z0 = overscan,
# pulse = strength of the punch-zooms on 808 + open-hat hits, tri = strength of the triple step/crash.
CAM = {0: dict(e=0.85, comp=0.60, zoom=(0.00, 0.04), tilt=(0, 0), z0=1.10, pulse=1.0, tri=1.0),
       1: dict(e=1.10, comp=None, zoom=(0.03, 0.03), tilt=(0, 0), z0=1.10, pulse=1.0, tri=1.0),
       2: dict(e=0.30, comp=0.40, zoom=(0.02, 0.03), tilt=(0, 0), z0=1.08, pulse=0.0, tri=0.0),    # as v4 until the drop
       3: dict(e=0.75, comp=0.58, zoom=(0.05, 0.01), tilt=(55, -45), z0=1.10, pulse=1.0, tri=1.0),  # tilt up the stairs
       4: dict(e=1.25, comp=0.60, zoom=(0.00, 0.05), tilt=(0, 0), z0=1.10, pulse=1.2, tri=1.1),     # dancer
       5: dict(e=0.60, comp=0.52, zoom=(0.00, 0.09), tilt=(-15, 10), z0=1.10, pulse=0.8, tri=0.8),  # close-up
       6: dict(e=0.95, comp=0.62, zoom=(0.03, 0.06), tilt=(-30, 15), z0=1.10, pulse=1.0, tri=1.0),
       7: dict(e=1.25, comp=0.62, zoom=(0.00, 0.03), tilt=(0, 0), z0=1.10, pulse=1.2, tri=1.0)}
DROP_E, DROP_COMP, DROP_Z = 1.15, 0.45, 0.30   # shot 3 after the drop


def _visible_h(i):
    return H - 2 * BAR * ease(i / 24)          # the letterbox closes over the first 24 frames


def _pulse_env(q):                             # snap in over 2 frames, fall back fast
    return 0.6 if q < 1 else (1.0 if q < 2 else np.exp(-(q - 1) / 2.6))


class Camera:
    def __init__(self, singer_cx):
        self.vx, self.vy, self.z, self.r = (np.zeros(N) for _ in range(4))
        trk = 0.0
        hit_dir = np.random.default_rng(7).uniform(0, 2 * np.pi, len(JOINT_F) + 64)    # each jolt its own direction
        hit_idx = {f: n for n, f in enumerate(JOINT_F)}
        for k, (s0, s1) in enumerate(SHOTS):
            st = CAM[k]
            rng = np.random.default_rng(100 + k)                     # each shot: its own operator
            f = rng.uniform(0.18, 1.3, (3, 4)); ph = rng.uniform(0, 2 * np.pi, (3, 4)); am = rng.uniform(0.5, 1, (3, 4))
            fj = rng.uniform(3.5, 6.0, 2); pj = rng.uniform(0, 2 * np.pi, 2)
            for i in range(s0, s1):
                t = (i - s0) / FPS
                u = (min(i, FREEZE) - s0) / max(1, s1 - s0 - 1)
                e, pulse_k, tri_k, comp = st["e"], st["pulse"], st["tri"], st["comp"]
                wd = 0.0
                if k == 2:                                           # the drop: from v4 framing to high energy
                    wd = ease((i - (DROP_F - 2)) / 5)                # whip passes through the hit, settles 3 frames later
                    e = e * (1 - wd) + DROP_E * wd
                    pulse_k = 1.2 if i > DROP_F else 0.0
                e *= (0.35 if i >= FREEZE else 1.0)
                n3 = [(am[c] * np.sin(2 * np.pi * f[c] * t + ph[c])).sum() / am[c].sum() for c in range(3)]
                hx = e * (21 * n3[0] + 2.6 * np.sin(2 * np.pi * fj[0] * t + pj[0]))      # screen px
                hy = e * (14 * n3[1] + 2.0 * np.sin(2 * np.pi * fj[1] * t + pj[1]))
                rot = e * 0.65 * n3[2]                                                     # degrees
                past = BEAT_F[BEAT_F <= i]                                                # bob on every beat
                if len(past) and past[-1] >= s0:
                    hy += e * 4 * np.exp(-(i - past[-1]) / 4)
                z = st["z0"] + st["zoom"][0] + (st["zoom"][1] - st["zoom"][0]) * ease(u)
                if i < FREEZE:
                    # punch-zoom + jolt on every 808 + open-hat hit
                    for hf in JOINT_F:
                        q = i - hf
                        if 0 <= q < 12 and hf >= s0 + 3 and hf not in TRIP_HITS and pulse_k > 0:
                            env = _pulse_env(q); a = hit_dir[hit_idx[hf]]
                            z += 0.055 * pulse_k * env
                            hx += 14 * pulse_k * np.cos(a) * np.exp(-q / 2.5)
                            hy += 10 * pulse_k * np.sin(a) * np.exp(-q / 2.5)
                            rot += 0.35 * pulse_k * np.sign(np.sin(3 * a)) * np.exp(-q / 3)
                    # the triple: step, step, crash, then release
                    tz = 0.0
                    for g in TRIP_F:
                        if not (s0 <= g[0] < s1) or tri_k <= 0:
                            continue
                        f1, f2, f3 = g
                        v = (0.07 * ease((i - f1 + 1) / 2) + 0.07 * ease((i - f2 + 1) / 2)
                             + 0.16 * ease((i - f3 + 1) / 2)) * (1 - ease((i - (f3 + 9)) / 12))
                        tz += v * tri_k
                        for hf, amp in ((f1, 16), (f2, 20), (f3, 30)):
                            q = i - hf
                            if 0 <= q < 12:
                                a = hit_dir[hit_idx.get(hf, 0) + 3]
                                hx += amp * tri_k * np.cos(a) * np.exp(-q / 2.5)
                                hy += 0.7 * amp * tri_k * np.sin(a) * np.exp(-q / 2.5)
                                rot += 0.5 * tri_k * np.sign(np.cos(5 * a)) * np.exp(-q / 3)
                    z += min(tz, 0.34)
                # tracking: follow the singer with an operator's lag (restarts at each cut)
                target = 0.0 if comp is None else 0.75 * (singer_cx[i] - comp * W)
                rate = 0.10
                if k == 2 and wd > 0:                                # zoom-in whip onto the singer on the drop
                    target = target * (1 - wd) + 0.75 * (singer_cx[i] - DROP_COMP * W) * wd
                    zd = ease((i - (DROP_F - 2)) / 2)                # the zoom lands exactly on the 808 frame
                    z += DROP_Z * zd - 0.12 * ease((i - DROP_F - 10) / 60)
                    rate = 0.45 if abs(i - DROP_F) < 5 else 0.12
                    q = i - DROP_F                                   # impact jolt on the hit
                    if 0 <= q < 12:
                        hx += 30 * np.exp(-q / 2.5); hy -= 18 * np.exp(-q / 2.5); rot += 0.6 * np.exp(-q / 3)
                if k == 6:                                           # 808 break: creep in, then crash + jolt on its return
                    bs, q = int(round(40.34 * FPS)), i - WHIP_F
                    release = 1 - ease((q - 10) / 14)
                    z += (0.10 * ease((i - bs) / (WHIP_F - bs)) + 0.20 * ease((q + 1) / 2)) * release
                    if 0 <= q < 12:
                        hx -= 32 * np.exp(-q / 2.5); hy += 20 * np.exp(-q / 2.5); rot -= 0.6 * np.exp(-q / 3)
                trk = target if i == s0 else trk + (target - trk) * rate
                tilt = st["tilt"][0] + (st["tilt"][1] - st["tilt"][0]) * ease(u)
                vx = W / 2 + trk + hx / z
                vy = H / 2 + tilt + hy / z
                # keep the view inside the picture (rotation needs a little margin)
                m = 6 + abs(np.sin(np.radians(rot))) * W / 2
                hw, hh = W / (2 * z), _visible_h(i) / (2 * z)
                self.vx[i] = np.clip(vx, hw + m, W - hw - m)
                self.vy[i] = np.clip(vy, hh + m, H - hh - m)
                self.z[i], self.r[i] = z, rot

    def matrix(self, i):
        c, s = np.cos(np.radians(self.r[i])), np.sin(np.radians(self.r[i]))
        z, vx, vy = self.z[i], self.vx[i], self.vy[i]
        return np.float32([[c / z, s / z, vx - (c * W / 2 + s * H / 2) / z],
                           [-s / z, c / z, vy - (-s * W / 2 + c * H / 2) / z]])

    def warp_quarter(self, mask, i):
        """Move a quarter-res source-space mask into the camera's view."""
        M = self.matrix(i).copy(); M[:, 2] /= 4
        return cv2.warpAffine(mask, M, (W // 4, H // 4), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _warp(self, img, vx, vy, z, rot):
        c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        M = np.float32([[c / z, s / z, vx - (c * W / 2 + s * H / 2) / z],
                        [-s / z, c / z, vy - (-s * W / 2 + c * H / 2) / z]])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)

    def apply(self, img, i):
        vx, vy, z, r = self.vx[i], self.vy[i], self.z[i], self.r[i]
        if i in CUT_F or i == 0:
            return self._warp(img, vx, vy, z, r)
        pz = self.z[i - 1]
        if abs(z / pz - 1) > 0.008:            # fast zoom: blend a few in-between zooms (radial motion blur)
            out = sum(self._warp(img, vx, vy, pz + (z - pz) * a, r) for a in (0.25, 0.5, 0.75, 1.0)) / 4
        else:
            out = self._warp(img, vx, vy, z, r)
        dx, dy = (vx - self.vx[i - 1]) * z, (vy - self.vy[i - 1]) * z   # screen speed, px/frame
        if abs(dx) > 4:
            out = cv2.blur(out, (min(161, int(abs(dx) * 0.9)) | 1, 1))
        if abs(dy) > 4:
            out = cv2.blur(out, (1, min(121, int(abs(dy) * 0.9)) | 1))
        return out


# ------------------------------------------------------------------ look pieces
STOPS = np.array([[6, 4, 3], [150, 128, 106], [244, 247, 250]], np.float32)   # BGR near-black, steel, near-white (v8: real blacks again)
LOOK_GAIN = 1.12
LUT = np.zeros((256, 3), np.float32)
for v in range(256):
    t = np.clip((v / 255 - 0.035) / 0.965, 0, 1) ** 0.85
    t = t * t * (3 - 2 * t) * 0.40 + t * 0.60
    LUT[v] = STOPS[0] + (STOPS[1] - STOPS[0]) * (t / 0.5) if t < 0.5 else STOPS[1] + (STOPS[2] - STOPS[1]) * ((t - 0.5) / 0.5)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
LUT = np.clip(LUT * LOOK_GAIN, 0, 255)
VIGN = (1 - 0.12 * (((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2) / 2)[..., None]

PUSH = {0: (0.0, 0.06), 3: (0.0, 0.05), 4: (0.06, 0.0), 5: (0.0, 0.07), 6: (0.0, 0.04), 7: (0.0, 0.03)}
PUSH_CY = H * 0.45


def push_amount(i):
    s = shot_of(i)
    if s not in PUSH:
        return 0.0
    s0, s1 = SHOTS[s]
    u = (min(i, FREEZE) - s0) / max(1, s1 - s0 - 1)
    a0, a1 = PUSH[s]
    return a0 + (a1 - a0) * ease(u)


def push_maps(n, amount):
    s = 1 + amount * (0.35 + 0.65 * n)           # near pixels move more: real parallax
    cx = W / 2
    return (cx + (XX - cx) / s).astype(np.float32), (PUSH_CY + (YY - PUSH_CY) / s).astype(np.float32)


def remap(img, mx, my):
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def soft_blur(img, sigma):
    """Large Gaussian blur, computed at quarter size (fast, looks the same)."""
    if sigma < 0.5:
        return img
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 4), (W, H), interpolation=cv2.INTER_LINEAR)


def focus(g, n, f, strength):
    b = np.clip(np.abs(n - f) * strength, 0, 3)
    levels = [g] + [cv2.GaussianBlur(g, (0, 0), s) for s in (3.2, 6.8, 12.0)]
    lo = np.floor(b).astype(int); fr = b - lo
    out = np.zeros_like(g)
    for k in range(4):
        out += levels[k] * ((lo == k) * (1 - fr) + (lo + 1 == k) * fr)
    return out


def isoline(n, level, width_px=2.4):
    ns = cv2.GaussianBlur(n, (0, 0), 3.0)
    gy, gx = np.gradient(ns)
    fw = np.sqrt(gx * gx + gy * gy) * width_px + 1e-6
    return np.clip(1 - np.abs(ns - level) / fw, 0, 1)[..., None]


def grade(g, p):
    small = cv2.resize(np.clip((g / 255 - 0.72) / 0.28, 0, 1), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hl = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    out = LUT[np.clip(g, 0, 255).astype(np.uint8)]
    return out * (1 + 0.03 * p) + np.array([150, 190, 255], np.float32) * (0.16 + 0.12 * p) * hl


def colour(c):
    grey = c.mean(axis=2, keepdims=True)
    return (grey + 0.85 * (c - grey)) * 0.95


def finish(img, i):
    rng = np.random.default_rng(i)
    out = img * VIGN + rng.normal(0, 0.014 * 255, (H, W))[..., None]
    fade = 1 - ease((i - (N - 11)) / 10)            # short fade to black on the last frames
    return np.clip(out * fade, 0, 255)


# ------------------------------------------------------------------ transitions
def shift_blur(img, dx, length):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    L = int(length) | 1
    return cv2.blur(out, (L, 1)) if L > 2 else out


def transition(kind, k, L, new, old, n_new):
    """new/old: graded BGR float frames (old = last frame of the previous shot, held)."""
    u = (k + 1) / L
    if kind in ("sweep", "far_first"):
        v = ease(u)
        if kind == "sweep":                          # nearest-first: plane travels near -> far
            p = 1.0 - 1.05 * v
            m = (n_new > p)
        else:                                        # back of the room first: plane travels far -> near
            p = -0.05 + 1.1 * v
            m = (n_new < p)
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)[..., None]
        out = new * m + old * (1 - m)
        if 0 < v < 1:
            ln = isoline(n_new, p); out = out * (1 - ln) + WARM_LINE * ln
        return out
    if kind == "whip":                               # slide + motion blur, the switch hidden at peak blur
        blur = np.sin(np.pi * u) * 220
        if u < 0.5:
            return shift_blur(old, -ease(u / 0.5) * W * 0.30, blur)
        return shift_blur(new, (1 - ease((u - 0.5) / 0.5)) * W * 0.30, blur)
    if kind == "depth_dissolve":                     # soft crossfade, near things arrive first, no line
        v = ease(u)
        a = np.clip(v * 1.6 - (1 - n_new) * 0.6, 0, 1).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 3.0)[..., None]
        return new * a + old * (1 - a)
    if kind == "defocus":                            # old shot blurs out, new shot resolves from blur
        sig = 22 * np.sin(np.pi * u)
        mix = np.clip((u - 0.35) / 0.3, 0, 1)
        return soft_blur(old, sig) * (1 - mix) + soft_blur(new, sig) * mix
    if kind == "scanline":                           # a scan line wipes top -> bottom
        y = int(round(ease(u) * (H + 6))) - 3
        out = old.copy()
        if y > 0:
            out[:min(y, H)] = new[:min(y, H)]
        if 0 <= y < H:
            band = np.zeros((H, W), np.float32)
            band[max(0, y - 1):y + 2] = 1
            glow = cv2.GaussianBlur(band, (0, 0), 6)[..., None]
            out = out * (1 - band[..., None]) + WARM_LINE * band[..., None] + WARM_LINE * 0.25 * glow
        return out
    if kind == "punch":                              # hard cut that lands at 108% and settles
        s = 1 + 0.08 * (1 - ease(u))
        M = cv2.getRotationMatrix2D((W / 2, PUSH_CY), 0, s)
        return cv2.warpAffine(new, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(kind)


# ------------------------------------------------------------------ title: blackletter, on a depth layer
def title_font_path(arg):
    for p in (arg, HERE / "fonts" / "UnifrakturMaguntia-Book.ttf"):
        if p and Path(p).exists():
            return str(p), 0
    return "/System/Library/Fonts/HelveticaNeue.ttc", 9          # stand-in until the blackletter is installed


def render_title(lines, font_spec, max_w):
    """RGBA (BGR+alpha) sprite of the title: icy fill, thin navy outline. Fitted to max_w px."""
    size = 400
    for _ in range(2):
        f = ImageFont.truetype(font_spec[0], size, index=font_spec[1])
        widest = max(f.getlength(l) for l in lines)
        size = int(size * max_w / widest)
    f = ImageFont.truetype(font_spec[0], size, index=font_spec[1])
    stroke = max(2, int(size * 0.016))
    line_h = int(size * 0.92)
    wd = int(max(f.getlength(l) for l in lines)) + 6 * stroke
    ht = line_h * len(lines) + int(size * 0.45)
    img = Image.new("RGBA", (wd, ht), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k, l in enumerate(lines):
        d.text((wd / 2, int(size * 0.2) + k * line_h), l, font=f, anchor="mt", fill=ICE + (255,),
               stroke_width=stroke, stroke_fill=NAVY + (255,))
    a = np.array(img).astype(np.float32)   # PIL gives RGBA; we filled with BGR tuples, so channels are BGR already
    return a


# Per-format title layout. x = centre ("crop" = centre of the tracked crop), y = centre as a
# fraction of height, occ = how fully nearer things hide the letters (1 = solid; below 1 the
# letters stay faintly visible through the person, a ghost occlusion that keeps the word
# readable when the crop centres the artist right on the title).
# The Fraktur font draws "-" as a double stroke (reads as "="); its en dash is the clean single stroke.
SHOWN = TITLE.replace("-", "–")
LAYOUT = {"master": dict(lines=[SHOWN], width=W * 0.56, x=W * 0.385, y=0.46, occ=1.0),
          "9x16": dict(lines=["Rendez–", "Vous"], width=608 * 0.86, x="crop", y=0.30, occ=0.6),
          "1x1": dict(lines=[SHOWN], width=1080 * 0.86, x="crop", y=0.40, occ=0.6)}


class Title:
    def __init__(self, font_spec):
        self.sprites = {f: render_title(L["lines"], font_spec, L["width"]) for f, L in LAYOUT.items()}

    def alpha(self, i):
        return ease((i - TITLE_IN) / 20) * (1 - ease((i - (TITLE_OUT - 16)) / 16))

    def layer(self, fmt, i, cx, n):
        """Full-frame BGR colour + alpha for this format and frame, with depth occlusion."""
        a_t = self.alpha(i)
        if a_t <= 0.003:
            return None
        spr = self.sprites[fmt]
        sh, sw = spr.shape[:2]
        L = LAYOUT[fmt]
        x_c = cx if L["x"] == "crop" else L["x"]
        y_c = H * L["y"]
        drift = (1 - ease((i - TITLE_IN) / 30)) * 14      # settles upward into place
        s = 1 + push_amount(i) * (0.35 + 0.65 * TITLE_DEPTH)   # moves with the camera at its own depth
        # map sprite -> frame: scale about the push centre, like the scene at that depth
        tx = x_c - sw / 2
        ty = y_c - sh / 2 + drift
        M = np.float32([[s, 0, (tx - W / 2) * s + W / 2], [0, s, (ty - PUSH_CY) * s + PUSH_CY]])
        lay = cv2.warpAffine(spr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        col, al = lay[..., :3], lay[..., 3:] / 255
        # reveal: soft wipe from the top while fading in
        top = ty + sh * (-0.1 + 1.2 * ease((i - TITLE_IN) / 22))
        wipe = np.clip((top - YY[..., None]) / 40 + 0.5, 0, 1)
        # occlusion: anything nearer than the title plane passes in front of the letters
        occ = cv2.GaussianBlur((n > TITLE_DEPTH).astype(np.float32), (0, 0), 1.6)[..., None]
        al = al * wipe * (1 - L["occ"] * occ) * a_t
        return col, al


def apply_title(F, lay):
    col, al = lay
    glow = cv2.GaussianBlur(al[..., 0], (0, 0), 9)[..., None]
    return F * (1 - al) + col * al + np.array(ICE, np.float32) * 0.10 * glow


# ------------------------------------------------------------------ HUD (master only, in the letterbox bar)
_fonts = {}


def hud(img, i):
    a = ease((i - 166) / 15) * (1 - ease((i - 1146) / 10))
    if a <= 0.003:
        return img
    key = MONO
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(MONO[0], 14, index=0)
    f = _fonts[key]
    pil = Image.fromarray(img)
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    yb = H - BAR / 2
    fill = (244, 234, 220, int(255 * 0.42 * a))
    d.text((40, yb), f"SHOT {shot_of(i) + 1} / 8", font=f, fill=fill, anchor="lm")
    d.text((W - 40, yb), f"TC {tc(i)}", font=f, fill=fill, anchor="rm")
    return np.array(Image.alpha_composite(pil.convert("RGBA"), layer).convert("RGB"))


# ------------------------------------------------------------------ light: long-exposure trails, anamorphic streaks, diffusion
TRAIL_DECAY = 0.93                                  # per frame (half-life ~0.33 s); the trail also spreads every frame
TRAIL_TINT = np.array([255, 238, 220], np.float32)  # BGR icy white
STREAK_TINT = np.array([255, 196, 132], np.float32) # BGR anamorphic blue
BLOOM_TINT = np.array([226, 238, 250], np.float32)  # BGR warm white


def light_sources(c):
    """The studio's real lights, found in the colour footage: small, very bright spots (quarter res, 0-1).
    Large bright areas (a lit wall, a panel) are left out so streaks only come from point sources."""
    q = cv2.resize(c, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    L = np.clip((0.114 * q[..., 0] + 0.587 * q[..., 1] + 0.299 * q[..., 2] - 215) / 35, 0, 1)
    big = cv2.GaussianBlur(L, (0, 0), 10) > 0.35
    return (L * ~big).astype(np.float32)


class LightFX:
    """Stateful per output: highlights leave a decaying, spreading trail (long exposure); real lights get
    horizontal anamorphic streaks and bloom; highlights get a soft mist. Resets at every cut."""
    def __init__(self):
        self.trail, self.last = None, -9

    def apply(self, img, lights, i):
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hl = np.clip((small.mean(axis=2) / 255 - 0.36) / 0.5, 0, 1) ** 1.3
        if self.trail is None or i - self.last > 4 or i <= self.last or i in CUT_F:
            self.trail = hl.copy()
        else:
            self.trail = cv2.GaussianBlur(np.maximum(self.trail * TRAIL_DECAY ** (i - self.last), hl), (0, 0), 1.3)
        self.last = i
        drag = np.clip(self.trail - cv2.GaussianBlur(hl, (0, 0), 1.0), 0, 1)      # only the dragged-out light
        add = (0.85 * drag + 0.50 * cv2.GaussianBlur(drag, (0, 0), 5))[..., None] * TRAIL_TINT
        add = add + (0.04 * cv2.GaussianBlur(hl, (0, 0), 10))[..., None] * BLOOM_TINT    # mist diffusion
        add = cv2.resize(np.clip(add, 0, 255), (W, H), interpolation=cv2.INTER_LINEAR)
        return 255 - (255 - np.clip(img, 0, 255)) * (1 - add / 255)                # screen blend: never clips


# ------------------------------------------------------------------ flares v2: built per light, like a real anamorphic lens
class Flares:
    """For each strong real light: a hot core, a textured anamorphic streak with chromatic fringes and double
    lines, a six-blade starburst and veiling glare near the centre. No lens-ghost dots. Rendered at half
    resolution; each light keeps its texture."""
    HW, HH = W // 2, H // 2
    GHOSTS = [(-0.45, 9, (150, 255, 190), 0.10, "disc"), (0.35, 20, (255, 170, 230), 0.07, "ring"),
              (0.75, 7, (120, 210, 255), 0.12, "disc"), (1.20, 34, (255, 220, 160), 0.05, "ring"),
              (1.65, 14, (200, 255, 255), 0.06, "disc")]

    def __init__(self):
        self.xx = np.arange(self.HW, dtype=np.float32)

    def _tex(self, key, n, t):
        rng = np.random.default_rng(key)
        base = np.convolve(rng.normal(0, 1, n + 64), np.ones(9) / 9, "same")[:n]
        drift = np.roll(base, int(t * 3) % n)                     # the grain of the streak slowly crawls
        return 0.72 + 0.28 * (drift - drift.min()) / (np.ptp(drift) + 1e-6)

    def render(self, lights, cx, cy, i, roll=0.0):
        if lights is None or lights.max() < 0.25:
            return None
        nl, lab, st, cen = cv2.connectedComponentsWithStats((lights > 0.25).astype(np.uint8), 8)
        cand = []
        for k in range(1, nl):
            area = st[k, cv2.CC_STAT_AREA]
            m = lights[lab == k].mean()
            cand.append((area * m, cen[k][0] * 2, cen[k][1] * 2))   # quarter -> half res
        cand.sort(reverse=True)
        layer = np.zeros((self.HH, self.HW, 3), np.float32)
        oc = np.array([cx / 2, cy / 2])
        pk = 1 + 0.7 * pulse(i)
        for n_, (strength, px, py) in enumerate(cand[:4]):          # only the strongest lights flare
            s = min(1.0, strength / 40) * pk
            if s < 0.2:
                continue                                              # tiny lights would only make specks
            key = int(round(px / 40)) * 1000 + int(round(py / 40))
            flick = 0.9 + 0.1 * np.random.default_rng(key * 7919 + i).random()
            s *= flick
            # anamorphic streak: exponential falloff, chromatic layers, double lines, crawling texture
            L = 260 * (0.5 + s)
            fall = np.exp(-np.abs(self.xx - px) / L) * self._tex(key, self.HW, i)
            y0, y1 = int(max(0, py - 14)), int(min(self.HH, py + 15))
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            for ch, (wid, ln, amp) in enumerate(((1.6, 1.0, 1.0), (1.0, 0.8, 0.75), (0.7, 0.6, 0.55))):  # B, G, R
                core = np.exp(-((yy - py) / wid) ** 2) * (fall ** (1 / ln))
                dbl = 0.28 * (np.exp(-((yy - py - 4) / (wid * 0.8)) ** 2) + np.exp(-((yy - py + 4) / (wid * 0.8)) ** 2)) * fall ** 1.6
                layer[y0:y1, :, ch] += (core + dbl) * amp * s * 255 * 0.9
            # local patch: hot core + six-blade starburst + wide glow
            R = int(90 * (0.5 + s))
            xa, xb, ya, yb = int(max(0, px - R)), int(min(self.HW, px + R)), int(max(0, py - R)), int(min(self.HH, py + R))
            if xb > xa and yb > ya:
                gx, gy = np.meshgrid(np.arange(xa, xb) - px, np.arange(ya, yb) - py)
                rr = np.sqrt(gx ** 2 + gy ** 2) + 1e-3
                th = np.arctan2(gy, gx) - np.radians(roll) + 0.3
                star = np.exp(-rr / (R * 0.35)) * np.maximum(np.cos(3 * th), 0) ** 60     # six thin rays
                core = np.exp(-(rr / 2.2) ** 2) + 0.35 * np.exp(-(rr / 14) ** 2) + 0.08 * np.exp(-(rr / 45) ** 2)
                patch = (core + 0.45 * star)[..., None] * np.array([235, 242, 250], np.float32) * s
                layer[ya:yb, xa:xb] += patch
            # veiling glare when a strong light is near the centre
            dist = np.hypot(px - oc[0], py - oc[1]) / self.HW
            if dist < 0.3 and s > 0.4:
                layer += np.array([200, 215, 230], np.float32) * 0.05 * s * (1 - dist / 0.3)
        return cv2.resize(np.clip(layer, 0, 255), (W, H), interpolation=cv2.INTER_LINEAR)


def screen(img, add):
    return 255 - (255 - np.clip(img, 0, 255)) * (1 - np.clip(add, 0, 255) / 255)


# ------------------------------------------------------------------ chaos, keyed to the track (all times measured)
# Colour punches: hard switch from ghost to the real footage on impacts, then straight back.
# (start frame, length, exit) - exit "cut" snaps back, "sweep" hands back to the ghost nearest-first.
PUNCH = [(101, 3, "cut"),                                  # downbeat 3.36 s
         (166, 2, "cut"), (179, 5, "cut"),                 # triple 5.1 s: hits 2 and 3
         (203, 3, "cut"),                                  # downbeat 6.78 s
         (411, 10, "sweep"),                               # THE DROP 13.70 s
         (512, 3, "cut"),                                  # downbeat 17.07 s
         (610, 2, "cut"), (616, 5, "cut"),                 # triple 20.1 s
         (744, 3, "cut"), (770, 3, "cut"),                 # strobe in the dancer shot
         (816, 2, "cut"), (822, 5, "cut"),                 # triple 27.0 s
         (951, 4, "cut"), (970, 4, "cut"),                 # double triple 31.2 s (third hits only, kept short)
         (1025, 3, "cut"),                                 # downbeat 34.17 s
         (1141, 5, "cut"),                                 # triple 37.7 s (third hit only: the cut follows closely)
         (1234, 9, "sweep"),                               # second drop 41.15 s
         (1334, 4, "cut")]                                 # downbeat 44.45 s
STUTTER = [(414, 444, 2), (731, 800, 3), (1334, 1372, 2)]   # step-printed frames: (start, end, hold)
GLITCH = [(387, 5), (1210, 5)]                             # the 808 drops out: signal glitch
IMPACT = {f: a for f, a in [(154, 0.6), (166, 0.8), (179, 1.0), (604, 0.6), (610, 0.8), (616, 1.0), (809, 0.6),
                            (816, 0.8), (822, 1.0), (951, 0.45), (970, 0.5), (1131, 0.4), (1136, 0.5), (1141, 1.0),
                            (411, 1.2), (1234, 1.2), (101, 0.5), (203, 0.5), (512, 0.5), (1025, 0.5), (1334, 0.6)]}
RING_HITS = [f for f in JOINT_F if not (319 <= f < DROP_F)]   # shot 3 stays calm until the drop

# Posterize time (v9): the finished frame (camera, light, everything) only updates on "key" frames and is
# held in between, like After Effects' Posterize Time. Base ~12 fps (holds 2,3,2,3), 10 fps (hold 3) in the
# most energetic stretches, 15 fps (hold 2) on the close-up. The grid re-anchors on every musical event so
# hits, punches and cuts still land on their exact frame.
POSTER_FAST = [(414, 444), (731, 800), (1334, 1372)]          # hold 3
POSTER_CLOSE = [(919, 1151)]                                  # hold 2


def poster_keys():
    events = set(CUT_F) | set(JOINT_F) | set(IMPACT) | {g for g, _ in GLITCH}
    events |= {st for st, _, _ in PUNCH} | {st + ln for st, ln, _ in PUNCH}
    is_key = np.zeros(N, bool)
    nxt, alt = 0, 0
    for i in range(N):
        if i in events or i >= nxt:
            is_key[i] = True
            if any(a <= i < b for a, b in POSTER_FAST):
                h = 3
            elif any(a <= i < b for a, b in POSTER_CLOSE):
                h = 2
            else:
                h = 2 + alt; alt ^= 1
            nxt = i + h
    return is_key


def punch_alpha(i, n):
    for st, ln, ex in PUNCH:
        if st <= i < st + ln:
            return 1.0
        if ex == "sweep" and st + ln <= i < st + ln + 4:      # ghost takes back over, nearest first
            p = 1.05 - 1.1 * ease((i - st - ln + 1) / 4)
            return cv2.GaussianBlur((n < p).astype(np.float32), (0, 0), 2.0)[..., None]
    return 0.0


def impact(i):
    return max([a * np.exp(-(i - f) / 3.0) for f, a in IMPACT.items() if 0 <= i - f < 12] + [0.0])


def punchy(c):
    """The real footage for impact moments: crushed blacks, bright highlights, a touch less saturation."""
    x = np.clip(c / 255, 0, 1)
    y = np.clip((x - 0.04) / 0.96, 0, 1) ** 0.85
    y = y * y * (3 - 2 * y) * 0.5 + y * 0.5
    grey = y.mean(axis=2, keepdims=True)
    return np.clip((grey + 0.9 * (y - grey)) * 255 * 1.12, 0, 255)


def glitch(img, i, k):
    rng = np.random.default_rng(5000 + i)
    out = img.copy()
    y = 0
    while y < H:
        h = int(rng.integers(8, 70))
        if rng.random() < 0.55:
            out[y:y + h] = np.roll(img[y:y + h], int(rng.integers(-90, 90)), axis=1)
        y += h
    out[..., 2] = np.roll(out[..., 2], 7, axis=1); out[..., 0] = np.roll(out[..., 0], -7, axis=1)
    return out * (0.85 if k % 2 else 1.0)


def aberration(img, a):
    """Radial chromatic aberration: red pushed out, blue pulled in, strength a (0-1+)."""
    if a < 0.05:
        return img
    out = img.copy()
    for ch, sc in ((2, 1 + 0.007 * a), (0, 1 - 0.007 * a)):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, sc)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return out


# ------------------------------------------------------------------ speed ramps (v10)
# Per shot: (first re-timed output frame, [(start, end, speed) in OUTPUT frames]). The free frames of each
# region absorb the difference so every shot still starts and ends on its own footage: cuts stay on the beat,
# the audio is untouched, the camera and all effects stay on the music's timeline. Only the footage re-times.
RAMPS = {0: (0, [(179, 214, 0.18), (218, 230, 3.6)]),        # v11 extreme: triple crash 5.97 s
         2: (387, [(387, 411, 3.0), (411, 451, 0.20)]),      # 808 break: 3x rush; THE DROP: 0.2x (shot 3 untouched before 12.9 s)
         3: (538, [(616, 650, 0.20), (654, 665, 3.5)]),      # triple crash 20.5 s
         4: (715, [(822, 860, 0.18), (864, 877, 3.4)]),      # dancer triple 27.4 s
         5: (919, [(1009, 1086, 0.25), (1095, 1118, 3.5)]),  # close-up colour bleed
         6: (1151, [(1220, 1234, 3.2), (1234, 1274, 0.20)]), # end of the second break, then slam
         7: (1318, [(1334, 1356, 0.25)])}                    # final downbeat before the freeze
# Each slow-mo starts on exactly the original footage: frames before a shot's first ramp play at normal
# speed, and the time a slow-mo borrows is paid back by the rush (and the free frames) after it.


def build_timemap():
    src = np.arange(N, dtype=np.float64)
    for k, (r0, segs) in RAMPS.items():
        s0, s1 = SHOTS[k]
        e = min(s1, FREEZE) if k == len(SHOTS) - 1 else s1
        L = e - r0
        v, fixed = np.ones(L), np.zeros(L, bool)
        fixed[:segs[0][0] - r0] = True                       # normal speed up to the first ramp: moments stay in place
        for a, b, sp in segs:
            v[a - r0:b - r0], fixed[a - r0:b - r0] = sp, True
        v[~fixed] = max(0.3, (L - v[fixed].sum()) / max(1, (~fixed).sum()))
        v = gauss1d(v, 1.5)                                  # ramps ease in and out of each speed
        cum = np.concatenate([[0.0], np.cumsum(v)[:-1]])
        src[r0:e] = r0 + cum * (L - 1) / cum[-1]             # exact fit: region covers its own footage
    return np.clip(src, 0, N - 1)


SRC_F = build_timemap()
SRC_IDX = np.floor(SRC_F + 1e-6).astype(int)
SRC_FRAC = SRC_F - SRC_IDX                                     # used to blend frames in slow-mo


# ------------------------------------------------------------------ the frame function
class Renderer:
    def __init__(self, singer_cx=None):
        self.scx = singer_cx
        self.src = cv2.VideoCapture(SRC)
        self.gh = cv2.VideoCapture(GHOST)
        self.old = {}          # graded last frame of the outgoing shot, for each transition
        self.frozen = None
        self.hold = None       # step-print stutter: the source frame being held
        self.pos, self.cache = -1, {}   # footage frames read so far (the time map only ever moves forward)

    def read_src(self, j):
        j = min(j, N - 1)
        while self.pos < j - 1:
            self.src.grab(); self.gh.grab(); self.pos += 1
        if self.pos < j:
            ok1, col = self.src.read(); ok2, gh = self.gh.read()
            if not (ok1 and ok2):
                raise SystemExit(f"ran out of footage at {j}")
            self.pos = j
            self.cache = {k: v for k, v in self.cache.items() if k >= j - 1}
            self.cache[j] = (col, gh)
        return self.cache[j]

    def skip(self, i):
        if i == FREEZE:                                 # previews: keep the freeze frame
            col, gh = self.read_src(SRC_IDX[i])
            self.frozen = (col, gh, depth(SRC_IDX[i]))

    def frame(self, i):
        """Returns (graded BGR float frame before grain/vignette, pushed depth, light sources)."""
        j = SRC_IDX[i]                                  # footage frame for this moment (speed ramps)
        col, gh = self.read_src(j)
        fr = SRC_FRAC[i]
        blend = fr > 0.05 and i + 1 < N and SRC_IDX[min(i + 1, N - 1)] - j <= 1 and shot_of(min(j + 1, N - 1)) == shot_of(j)
        if blend:                                       # slow-mo: blend with the next footage frame
            col1, gh1 = self.read_src(j + 1)
            col = cv2.addWeighted(col, 1 - fr, col1, fr, 0)
            gh = cv2.addWeighted(gh, 1 - fr, gh1, fr, 0)
        s = shot_of(i); s0, s1 = SHOTS[s]
        t = (i - s0) / FPS
        if i >= FREEZE:
            if self.frozen is None:
                self.frozen = (col, gh, depth(j))
            col, gh, n = self.frozen
        else:
            n = depth(j) if not blend else depth(j) * (1 - fr) + depth(j + 1) * fr
        for a_, b_, k_ in STUTTER:                      # step-print: hold the picture while time runs on
            if a_ <= i < b_:
                if (i - a_) % k_ == 0 or self.hold is None:
                    self.hold = (col, gh, n)
                else:
                    col, gh, n = self.hold
        g = cv2.cvtColor(gh, cv2.COLOR_BGR2GRAY).astype(np.float32)
        c = col.astype(np.float32)

        amt = push_amount(i)
        if amt > 1e-4:
            mx, my = push_maps(n, amt)
            g, c, n = remap(g, mx, my), remap(c, mx, my), remap(n, mx, my)

        f_face = 0.62
        if s == 5:                                      # shot 6: focus pull mic -> face
            f = 0.92 - (0.92 - f_face) * ease((t - 1.5) / 1.2)
            g = focus(g, n, f, 9.0)

        out = grade(g, pulse(i))
        if self.scx is not None and i < FREEZE:          # soft key light that follows the singer: lifts him and
            sx = self.scx[i]                              # things at his depth, never the background behind him
            sd = float(np.median(n[int(H * 0.3):int(H * 0.7), int(max(0, sx - 60)):int(min(W, sx + 60))]))
            band = np.clip(1 - np.abs(n - sd) / 0.22, 0, 1) * (n > 0.12)
            spot = np.exp(-((XX - sx) / (0.11 * W)) ** 2 - ((YY - H * 0.48) / (0.42 * H)) ** 2) * band
            out = out * (1 + 0.32 * cv2.GaussianBlur(spot, (0, 0), 6)[..., None])
        cg = colour(c)

        if i < 64:                                      # opening: scan from colour into ghost
            p = 1.05 - 1.15 * ease((i - 8) / 44)
            m = cv2.GaussianBlur((n > p).astype(np.float32), (0, 0), 2.0)[..., None]
            out = out * m + cg * (1 - m)
            if 0 < ease((i - 8) / 44) < 1:
                ln = isoline(n, p); out = out * (1 - ln) + WARM_LINE * ln

        if s == 5:                                      # shot 6: colour bleeds into the singer, then recedes
            a = ease((t - 3.0) / 0.8) * (1 - ease((t - 5.6) / 0.8))
            if a > 0:
                band = np.clip(1 - np.abs(n - f_face) / 0.14, 0, 1) ** 0.7 * (n > 0.40)
                m = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.5)[..., None] * a
                grey = c.mean(axis=2, keepdims=True)
                pop = np.clip((grey + 1.25 * (c - grey)) * 1.15, 0, 255)
                out = out * (1 - m) + pop * m

        if i >= FREEZE:                                 # ending: colour returns from the back of the room forwards
            v = ease((i - FREEZE) / 26)
            p = -0.05 + 1.1 * v
            m = cv2.GaussianBlur((n < p).astype(np.float32), (0, 0), 2.0)[..., None]
            out = out * (1 - m) + cg * m
            if 0 < v < 1:
                ln = isoline(n, p); out = out * (1 - ln) + WARM_LINE * ln

        # depth-scan rings: a contour races from near to far after each 808 + open-hat hit
        for hf in RING_HITS:
            q = i - hf
            if 0 <= q < 10:
                ln = isoline(n, 1.0 - 1.1 * ease((q + 1) / 10), 2.0) * (0.7 * (1 - q / 10) ** 1.2)
                out = out * (1 - ln) + np.array([255, 236, 205], np.float32) * ln

        pa = punch_alpha(i, n)                          # colour punch: switch to the real footage for impact
        if np.max(pa) > 0:
            out = out * (1 - pa) + punchy(c) * pa
        ex = impact(i)                                  # exposure kick on impacts
        if ex > 0.05:
            out = out * (1 + 0.16 * min(ex, 1.2))
        for gs, gl in GLITCH:                           # the 808 drops out: signal glitch
            if gs <= i < gs + gl:
                out = glitch(out, i, i - gs)

        for cut, (kind, L) in TRANSITIONS.items():
            if cut <= i < cut + L:
                out = transition(kind, i - cut, L, out, self.old.get(cut), n)
            if i == cut - 1:
                self.old[cut] = out.copy()
        return out, n, light_sources(c)


FORMATS = ("master", "9x16", "1x1")


def drop_zoom(i):
    """Phone formats (no handheld camera): a crop zoom that lands on the 808 at the drop (13.70 s), holds,
    and eases back just before the cut at 17.9 s, like the 16:9 camera."""
    zd = ease((i - (DROP_F - 2)) / 2)
    return 1 + (0.40 * zd - 0.15 * ease((i - DROP_F - 10) / 60)) * (1 - ease((i - 526) / 12))


def outputs(out, n, lights, i, cx, title, cam=None, fx=None):
    """Graded frame -> dict of uint8 RGB frames per format. fx: {'master': LightFX, 'src': LightFX}."""
    res = {}
    base = None
    for fmt in FORMATS:
        lay = title.layer(fmt, i, cx, n) if TITLE_ON else None
        if fmt == "master" and cam is not None:            # handheld camera on the main video only
            framed = cam.apply(apply_title(out, lay) if lay else out, i)
            if fx:                                         # light reacts in screen space: camera moves drag it too
                lq = cam.warp_quarter(lights, i)
                framed = fx["master"].apply(framed, lq, i)
                fl = fx["flares"].render(lq, W / 2, H / 2, i, cam.r[i])
                if fl is not None:
                    framed = screen(framed, fl)
                framed = aberration(framed, impact(i))
            F = finish(framed, i)
        elif lay is None:
            if base is None:
                if fx:
                    b = fx["src"].apply(out, lights, i)
                    fl = fx["flares"].render(lights, cx, H / 2, i)
                    b = aberration(screen(b, fl) if fl is not None else b, impact(i))
                else:
                    b = out
                base = finish(b, i)
            F = base
        else:
            F = finish(apply_title(out, lay), i)
        if fmt == "master":
            m = F.copy()
            b = int(round(BAR * ease(i / 24)))
            if b:
                m[:b] = (14, 10, 8); m[H - b:] = (14, 10, 8)
            res[fmt] = hud(cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_BGR2RGB), i)
        else:
            cw, size = (608, (1080, 1920)) if fmt == "9x16" else (1080, (1080, 1080))
            zc = drop_zoom(i)
            cw, ch = int(round(cw / zc)), int(round(H / zc))
            x0 = int(np.clip(round(cx - cw / 2), 0, W - cw))
            y0 = int(np.clip(round(H * 0.46 - ch / 2), 0, H - ch))
            crop = F[y0:y0 + ch, x0:x0 + cw]
            if crop.shape[1] != size[0] or crop.shape[0] != size[1]:
                crop = cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4)
            res[fmt] = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preview", help="comma list of frames: write stills only")
    ap.add_argument("--title-font")
    ap.add_argument("--formats", default="master,9x16,1x1", help="which deliverables to render")
    a = ap.parse_args()
    global FORMATS
    FORMATS = tuple(f for f in ("master", "9x16", "1x1") if f in a.formats.split(","))
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    fspec = title_font_path(a.title_font)
    print(f"title font: {fspec[0]}  formats: {', '.join(FORMATS)}", flush=True)
    title = Title(fspec)
    cx = track_singer()[SRC_IDX]                    # the singer's position in the re-timed footage
    cam = Camera(cx) if "master" in FORMATS else None
    fx = {"master": LightFX(), "src": LightFX(), "flares": Flares()}
    r = Renderer(cx)
    if a.preview:
        want = sorted(int(x) for x in a.preview.split(","))
        needed = set(want) | {c - 1 for c in TRANSITIONS} | {w - k for w in want for k in range(1, 16) if w - k >= 0}   # trails need warm-up
        for i in range(max(want) + 1):
            if i not in needed:
                r.skip(i); continue
            o, n, lt = r.frame(i)
            fr = outputs(o, n, lt, i, cx[i], title, cam, fx)
            if i in want:
                for fmt, img in fr.items():
                    cv2.imwrite(str(out / f"preview_{i:04d}_{fmt}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("previews done", flush=True)
        return
    enc = {}
    specs = {"master": ((W, H), 16), "9x16": ((1080, 1920), 18), "1x1": ((1080, 1080), 18)}
    for fmt, ((w, h), crf) in specs.items():
        if fmt not in FORMATS:
            continue
        enc[fmt] = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-i", AUDIO, "-map", "0:v", "-map", "1:a",
                                     "-t", f"{N / FPS:.6f}", "-af", f"afade=t=out:st={N / FPS - 0.35:.3f}:d=0.35",   # full level; only the last 0.35 s fades with the picture
                                     "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                                     "-pix_fmt", "yuv420p", "-c:a", "aac_at", "-b:a", "320k", "-movflags", "+faststart",
                                     str(out / f"RENDEZVOUS_scan_cut_{fmt}.mp4")], stdin=subprocess.PIPE)
    t0 = time.time()
    keys = poster_keys()
    print(f"posterize time: {keys.sum()} key frames of {N} (~{keys.sum() / N * FPS:.1f} fps on average)", flush=True)
    held = None
    for i in range(N):
        o, n, lt = r.frame(i)                       # every frame is still read and graded (state stays exact)
        if keys[i] or held is None:
            held = {fmt: np.ascontiguousarray(img).tobytes() for fmt, img in outputs(o, n, lt, i, cx[i], title, cam, fx).items()}
        for fmt, buf in held.items():
            enc[fmt].stdin.write(buf)
        if (i + 1) % 100 == 0 or i + 1 == N:
            el = time.time() - t0
            print(f"frame {i + 1}/{N}  {el / (i + 1):.2f}s/frame  ~{el / (i + 1) * (N - i - 1) / 60:.1f} min left", flush=True)
    for e in enc.values():
        e.stdin.close(); e.wait()
    print("done", flush=True)


if __name__ == "__main__":
    main()
