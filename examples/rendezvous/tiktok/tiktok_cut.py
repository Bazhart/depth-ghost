#!/usr/bin/env python3
"""
RENDËZVOUS — TikTok cuts: ten 9:16 clips (1080x1920, 30 fps) in the look of the Remix (16:9 HQ), each on its
own moment of "holy water remake" (new mix). Clips 02 and 05 loop seamlessly: 8 bars of the hook, the audio
seam is crossfaded into what really precedes the first downbeat, and the footage, the camera and the light
trails run straight through the seam.

The picture is composed in footage space (2058x1080) with the remix pieces (depth ghost grade, colour punches,
depth rings, glitches, colour bleed, loom, triptych), then a handheld vertical camera frames it, and light
trails, flares, transitions, grain and posterize time are applied in the vertical frame.

usage: python tiktok_cut.py OUT_DIR [--clips 01,05] [--preview 0,40,80] [--sheet]
"""
import argparse
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PR = HERE.parent
sys.path.insert(0, str(PR))
sys.path.insert(0, str(HERE))
import remix_cut as rc  # noqa: E402
import final_cut as fc  # noqa: E402
from final_cut import W, H, FPS, ease  # noqa: E402
from tiktok_edl import CLIPS  # noqa: E402

OW, OH = 1080, 1920
SR = 44100
WAV = Path(__import__("os").environ.get("DG_AUDIO", str(PR / "audio" / "track.wav")))
EV = json.load(open(HERE / "song_events.json"))
BAR, BEAT, B0 = EV["bar"], EV["beat"], EV["bar0"]
PERSON = PR / "frames" / "person"

# Per shot: base zoom (1 = full footage height), the singer's face height in the footage (fraction of H; every
# zoom keeps the face in the upper part of the frame, clear of TikTok's caption area), and where the singer
# sits across the vertical frame (0-1; a touch left of centre keeps him clear of TikTok's buttons).
SHOT_CAM = {"A": (1.10, 0.26, 0.47), "B": (1.25, 0.45, 0.50), "C": (1.10, 0.30, 0.48), "D": (1.55, 0.44, 0.48),
            "E": (1.10, 0.27, 0.47), "F": (1.08, 0.27, 0.50), "G": (1.10, 0.26, 0.47), "H": (1.10, 0.30, 0.48)}
FACE_AT = 0.36                                        # the face sits at 36 % of the frame height, whatever the zoom


def bar_t(b):
    return B0 + b * BAR


# ------------------------------------------------------------------ the clip: timeline, edit list, events
class Clip:
    def __init__(self, key, spec):
        self.key, self.spec, self.name = key, spec, spec["name"]
        self.b0, self.b1 = spec["bars"]
        self.loop = spec.get("loop", False)
        self.a0 = max(0.0, bar_t(self.b0))
        if self.loop:                                        # exact frame count: the audio is fitted to it
            dur = (self.b1 - self.b0) * BAR
            self.N = int(round(dur * FPS)); self.r = dur * FPS / self.N
        else:
            self.N = int(round((bar_t(self.b1) - self.a0) * FPS)); self.r = 1.0
        self.a1 = self.a0 + self.N * self.r / FPS
        self.edl = [dict(f=self.bf(b), shot=sh, src=float(s), speed=float(sp), opt=dict(o)) for b, sh, s, sp, o in spec["edl"]]
        if self.loop:                                        # first segment = the last one running on through the seam
            last = self.edl[-1]
            self.edl[0].update(shot=last["shot"], speed=last["speed"], src=last["src"] + last["speed"] * (self.N - last["f"]))
        self.starts = np.array([e["f"] for e in self.edl])
        assert self.starts[0] == 0 and np.all(np.diff(self.starts) > 0), f"{key}: edit list out of order"
        self.cuts = set(int(s) for s in self.starts[1:])
        tf = self.tf
        inwin = lambda t: self.a0 - 1e-3 <= t < self.a1 - 1e-3
        self.joint = sorted({tf(t) for t in EV["joint"] if inwin(t)})
        self.hits = sorted({tf(t) for t in EV["hits"] if inwin(t)})
        self.groups = [tuple(tf(t) for t in g) for g in EV["groups"] if inwin(g[0])]
        self.breaks = [(tf(a), tf(b)) for a, b in EV["breaks"] if b > self.a0 + 0.02 and a < self.a1 - 0.02]
        self.drops = [tf(b) for a, b in EV["breaks"] if self.a0 + 0.02 < b < self.a1 - 0.1]
        if self.loop:
            self.drops += [tf(b) for a, b in EV["breaks"] if abs(b - self.a1) < 0.03]   # the seam itself (frame N = 0)
        self.beats2 = [tf(bar_t(self.b0) + k * 2 * BEAT) for k in range(int((self.b1 - self.b0) * 2) + 1)]
        self._effects()
        self.srcj = np.array([self.src_index(i) for i in range(self.N)])
        self.cx = np.array([rc.TRACK[j] for j in self.srcj])
        self.old = {}

    # time maps
    def tf(self, t):
        return int(round((t - self.a0) * FPS / self.r))

    def bf(self, b):
        return 0 if b <= 0 else self.tf(bar_t(self.b0 + b))

    def wrap(self, evs):
        evs = list(evs)
        return sorted(set(evs + [e - self.N for e in evs] + [e + self.N for e in evs])) if self.loop else evs

    def seg_index(self, i):
        return int(np.searchsorted(self.starts, i, side="right") - 1)

    def seg(self, i):
        return self.edl[self.seg_index(i)]

    def seg_len(self, k):
        return (self.starts[k + 1] if k + 1 < len(self.edl) else self.N) - self.starts[k]

    def seg_t(self, i):
        """Frames since the segment began (the first segment of a loop continues the last one)."""
        k = self.seg_index(i); t = i - self.starts[k]
        if self.loop and k == 0:
            t += self.N - self.starts[-1]
        return k, t

    def src_t(self, i):
        k, t = self.seg_t(i)
        e = self.edl[k]; o = e["opt"]
        if self.loop and k == 0:
            t -= self.N - self.starts[-1]
        if "stutter" in o:
            t = (t // o["stutter"]) * o["stutter"]
        if "freeze" in o:
            t = min(t, o["freeze"])
        return k, t

    def src_pos(self, i):
        k, t = self.src_t(i)
        e = self.edl[k]
        return rc.src_pos(e["shot"], e["src"], e["speed"], t), e["shot"]

    def src_index(self, i):
        p, shot = self.src_pos(i)
        return int(np.floor(p))

    def in_break(self, i):
        return any(a <= i < b for a, b in self.breaks) or (self.loop and any(a <= i + self.N < b or a <= i - self.N < b for a, b in self.breaks))

    def calm(self, i):
        return self.seg(i)["opt"].get("calm", False) or self.in_break(i)

    # effects keyed to the music
    def _effects(self):
        cand = []                                            # (priority, start, length, exit)
        for d in self.drops:
            cand.append((0, d, 10, "sweep"))
        if self.spec.get("triplet_punch", True):
            for g in self.groups:
                cand.append((1, g[2], 4, "cut"))
        for b in self.spec.get("punch", []):
            cand.append((2, self.bf(b), 3, "cut"))
        acc = []
        for pr, st, ln, ex in sorted(cand):
            near = any(abs(st - a) < 15 or (self.loop and abs(abs(st - a) - self.N) < 15) for _, a, _, _ in acc)
            if not near and not self.seg(st % self.N)["opt"].get("no_punch") and "tri" not in self.seg(st % self.N)["opt"]:
                acc.append((pr, st, ln, ex))
        self.punch = sorted((st, ln, ex) for _, st, ln, ex in acc)
        imp = {}
        for d in self.drops:
            imp[d] = 1.2
        for g in self.groups:
            for f, a in zip(g, (0.6, 0.8, 1.0)):
                imp[f] = max(imp.get(f, 0), a)
        for st, ln, ex in self.punch:
            imp[st] = max(imp.get(st, 0), 0.5)
        self.impacts = imp
        self.glitch = [(a, 5) for a, b in self.breaks if a >= 2 or self.loop]
        self.rings = [f for f in self.joint if not self.in_break(f)]
        self.tg = {f for g in self.groups for f in g}

    def punch_alpha(self, i, n):
        for st, ln, ex in self.punch:
            for s in ((st, st - self.N, st + self.N) if self.loop else (st,)):
                if s <= i < s + ln:
                    return 1.0
                if ex == "sweep" and s + ln <= i < s + ln + 4:
                    p = 1.05 - 1.1 * ease((i - s - ln + 1) / 4)
                    return cv2.GaussianBlur((n < p).astype(np.float32), (0, 0), 2.0)[..., None]
        return 0.0

    def impact(self, i):
        v = 0.0
        for f, a in self.impacts.items():
            for s in ((f, f - self.N, f + self.N) if self.loop else (f,)):
                if 0 <= i - s < 12:
                    v = max(v, a * np.exp(-(i - s) / 3.0))
        return v

    def pulse(self, i):
        k, t = self.seg_t(i)
        return min(1.0, sum(np.exp(-(i - h) / 5.0) for h in self.wrap(self.hits) if i - t <= h <= i and i - h < 30))

    def scan_state(self, i):
        scans = [(self.bf(b), max(2, self.bf(b + L) - self.bf(b)), kind) for b, L, kind in self.spec.get("scan", [])]
        mode = "colour" if scans and scans[0][2] == "in" else "ghost"
        for f, L, kind in scans:
            if i < f:
                break
            if i < f + L:
                return kind, (i - f + 1) / L
            mode = "ghost" if kind == "in" else "colour"
        return mode, None

    def keys(self):
        """Posterize time: the finished frame updates on key frames only (about 12 fps), re-anchored on every
        musical event so cuts, hits and punches land on their exact frame."""
        ev = set(int(s) for s in self.starts) | set(self.joint) | {f % self.N for f in self.impacts}
        ev |= {g for g, _ in self.glitch} | {st % self.N for st, _, _ in self.punch} | {(st + ln) % self.N for st, ln, _ in self.punch}
        fast = [(self.bf(a), self.bf(b)) for a, b in self.spec.get("fast", [])]
        is_key = np.zeros(self.N, bool); nxt, alt = 0, 0
        for i in range(self.N):
            if i in ev or i >= nxt:
                is_key[i] = True
                o = self.seg(i)["opt"]
                h = o.get("hold") or (3 if any(a <= i < b for a, b in fast) else 2 + alt)
                alt ^= 1; nxt = i + h
        return is_key


# ------------------------------------------------------------------ masks
@lru_cache(maxsize=48)
def person_matte(j, shot):
    """Soft matte (footage space) of the singer: Apple Vision person segmentation (temporally smoothed),
    pulled slightly inside the silhouette; the close-up uses its depth band (head in, microphone out)."""
    if shot == "F":
        n = rc.load(j)[2]
        return cv2.GaussianBlur(rc.singer_band(n, rc.TRACK[j]), (0, 0), 3)
    p = PERSON / f"{j:06d}.png"
    if not p.exists():
        return np.zeros((H, W), np.float32)
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255


# ------------------------------------------------------------------ footage-space composition
def ghost9(g, n, j, shot, p, keylight=True):
    """The Remix ghost, unchanged (remix_cut.ghost with the clip's own glow pulse)."""
    if shot == "F":                                         # close-up: focus pull mic -> face, on its own time
        t = (j - rc.SHOT["F"][0]) / FPS
        g = fc.focus(g, n, 0.92 - 0.30 * ease((t - 1.5) / 1.2), 9.0)
    out = fc.grade(g, p)
    if keylight:
        sx = rc.TRACK[j]
        sd = float(np.median(n[int(H * 0.3):int(H * 0.7), int(max(0, sx - 60)):int(min(W, sx + 60))]))
        band = np.clip(1 - np.abs(n - sd) / 0.22, 0, 1) * (n > 0.12)
        spot = np.exp(-((fc.XX - sx) / (0.11 * W)) ** 2 - ((fc.YY - H * 0.48) / (0.42 * H)) ** 2) * band
        out = out * (1 + 0.32 * cv2.GaussianBlur(spot, (0, 0), 6)[..., None])
    return out


LAYER_FADE = (np.clip((H - fc.YY[:, :1] - 0.04 * H) / (0.34 * H), 0, 1) ** 1.6).astype(np.float32)   # placed layers


def seg_env(t, L, a=4, b=4):
    return ease((t + 1) / a) * (1 - ease((t - (L - b - 1)) / b))


def compose(clip, i):
    """Footage-space frame i: (BGR float, depth, colour plate, source frame index)."""
    k, t = clip.seg_t(i)
    e = clip.edl[k]; opt = e["opt"]; shot = e["shot"]
    L = clip.seg_len(k) + (clip.seg_len(0) if clip.loop and k == len(clip.edl) - 1 else 0)
    if clip.loop and k == 0:
        L = clip.seg_len(0) + clip.seg_len(len(clip.edl) - 1)
    p, _ = clip.src_pos(i)
    c, g, n, j = rc.load_f(p, shot)
    out = ghost9(g, n, j, shot, clip.pulse(i))
    cg = fc.colour(c)

    mode, u = clip.scan_state(i)
    if mode == "colour":
        out = cg.copy()
    elif mode == "in":                                      # the scanner turns the room into the ghost, near to far
        pl = 1.05 - 1.15 * ease(u)
        m = cv2.GaussianBlur((n > pl).astype(np.float32), (0, 0), 2.0)[..., None]
        out = out * m + cg * (1 - m)
        ln = fc.isoline(n, pl); out = out * (1 - ln) + fc.WARM_LINE * ln
    elif mode == "out":                                     # colour comes back from the back of the room forwards
        pl = -0.05 + 1.1 * ease(u)
        m = cv2.GaussianBlur((n < pl).astype(np.float32), (0, 0), 2.0)[..., None]
        out = out * (1 - m) + cg * m
        ln = fc.isoline(n, pl); out = out * (1 - ln) + fc.WARM_LINE * ln

    if opt.get("bleed"):                                    # colour bleeds slowly into the singer, then recedes
        a = seg_env(t, L, 12, 8) * 0.9
        m = cv2.GaussianBlur(person_matte(j, shot), (0, 0), 4)[..., None] * a
        out = out * (1 - m) + rc.pop(c) * m

    if opt.get("iso"):                                      # the singer in colour, the room stays ghost
        a = seg_env(t, L, 3, 3) * float(opt["iso"])
        m = person_matte(j, shot)[..., None] * a
        out = out * (1 - m) + rc.pop(c) * m

    if "loom" in opt:                                       # double exposure: the close-up face looms in the room
        sh, s_, sp, op, dx, sc = opt["loom"]
        lc, lg, ln_, lj = rc.load_f(rc.src_pos(sh, s_, sp, t), sh)
        lgr = ghost9(lg, ln_, lj, sh, clip.pulse(i), keylight=False)
        m = cv2.GaussianBlur(rc.singer_band(ln_, rc.TRACK[lj]), (0, 0), 6)
        m = m * LAYER_FADE                                  # dissolve well before the layer's own bottom edge
        cam = clip.cam
        ww = H / cam.z[i] * 9 / 16
        lgr, m = rc.place([lgr, m], rc.TRACK[lj], cam.vx[i] + dx * ww, sc)
        out = fc.screen(out, lgr * m[..., None] * op * seg_env(t, L, 8, 6))

    if mode != "colour" and not clip.calm(i) and shot != "F":   # depth-scan rings on the 808 + open-hat hits (never across a close-up face)
        for hf in clip.wrap(clip.rings):
            q = i - hf
            if 0 <= q < 10:
                ln = fc.isoline(n, 1.0 - 1.1 * ease((q + 1) / 10), 2.0) * (0.7 * (1 - q / 10) ** 1.2)
                out = out * (1 - ln) + np.array([255, 236, 205], np.float32) * ln
    pa = clip.punch_alpha(i, n)                             # colour punches on the impacts
    if np.max(pa) > 0:
        out = out * (1 - pa) + fc.punchy(c) * pa
    ex = clip.impact(i)
    if ex > 0.05:
        out = out * (1 + 0.16 * min(ex, 1.2))
    for gs, gl in clip.glitch:
        if gs <= i < gs + gl:
            out = fc.glitch(out, i, i - gs)
    return out, n, c, j


# ------------------------------------------------------------------ the vertical handheld camera
class Cam9:
    """Handheld operator for the vertical frame, keyed to the music: noise, beat bob, punch-zooms on the 808 +
    open-hat hits, step-step-crash on the triplets, a slam on every drop, a slow creep through 808 breaks, and
    tracking that keeps the singer in frame. Loops are solved on a doubled timeline so the camera at the end
    of the clip flows into its first frame."""
    def __init__(self, clip):
        N, loop = clip.N, clip.loop
        V = 2 * N if loop else N
        segs = []
        for c in ((0, 1) if loop else (0,)):
            for k, e in enumerate(clip.edl):
                if c == 1 and k == 0:
                    continue                                 # merged with the last segment of the first copy
                segs.append((e["f"] + c * N, k))
        vs = np.array([s for s, _ in segs])
        ev = (lambda xs: sorted(set(list(xs) + [x + N for x in xs]))) if loop else (lambda xs: list(xs))
        joint, beats = ev(clip.joint), ev(clip.beats2)
        groups = [g for g0 in clip.groups for g in ((g0, tuple(x + N for x in g0)) if loop else (g0,))]
        drops = ev(clip.drops)
        breaks = [b for b0 in clip.breaks for b in ((b0, (b0[0] + N, b0[1] + N)) if loop else (b0,))]
        tg = {f for g in groups for f in g}
        hit_dir = np.random.default_rng(int(clip.key) * 7 + 3).uniform(0, 2 * np.pi, 8192)
        self.vx, self.vy, self.z, self.r = (np.zeros(V) for _ in range(4))
        trk = 0.0
        for q_, (s0, k) in enumerate(segs):
            s1 = vs[q_ + 1] if q_ + 1 < len(segs) else V
            e = clip.edl[k]; o = e["opt"]; shot = e["shot"]
            z0, cy, comp = SHOT_CAM[shot]
            z0, cy, comp = o.get("z", z0), o.get("cy", cy), o.get("comp", comp)
            rng = np.random.default_rng(int(clip.key) * 1000 + 17 * k + 1)
            f = rng.uniform(0.18, 1.3, (3, 4)); ph = rng.uniform(0, 2 * np.pi, (3, 4)); am = rng.uniform(0.5, 1, (3, 4))
            fj = rng.uniform(3.5, 6.0, 2); pj = rng.uniform(0, 2 * np.pi, 2)
            drift = rng.choice([-1, 1]) * 0.035
            e0 = o.get("e", rc.ENERGY.get(shot, 1.0))
            for v in range(s0, s1):
                i = v % N
                t = (v - s0) / FPS; u = (v - s0) / max(1, s1 - s0 - 1)
                brk = any(a <= v < b for a, b in breaks)
                en = e0 * (0.55 if brk else 1.0)
                n3 = [(am[c] * np.sin(2 * np.pi * f[c] * t + ph[c])).sum() / am[c].sum() for c in range(3)]
                hx = en * (13 * n3[0] + 1.8 * np.sin(2 * np.pi * fj[0] * t + pj[0]))
                hy = en * (14 * n3[1] + 2.0 * np.sin(2 * np.pi * fj[1] * t + pj[1]))
                rot = en * 0.65 * n3[2]
                past = [b for b in beats if s0 <= b <= v]
                if past:
                    hy += en * 4 * np.exp(-(v - past[-1]) / 4)
                z = z0 + (drift * ease(u) if drift > 0 else -drift * (1 - ease(u)))
                calm = o.get("calm", False) or brk
                if not calm:
                    for hf in joint:
                        q = v - hf
                        if 0 <= q < 12 and hf not in tg:
                            env = fc._pulse_env(q); a = hit_dir[hf % 8192]
                            z += 0.05 * env
                            hx += 9 * np.cos(a) * np.exp(-q / 2.5); hy += 10 * np.sin(a) * np.exp(-q / 2.5)
                            rot += 0.35 * np.sign(np.sin(3 * a)) * np.exp(-q / 3)
                tz = 0.0
                for g in groups:
                    f1, f2, f3 = g
                    tz += (0.06 * ease((v - f1 + 1) / 2) + 0.06 * ease((v - f2 + 1) / 2)
                           + 0.15 * ease((v - f3 + 1) / 2)) * (1 - ease((v - (f3 + 9)) / 12))
                    for hf, amp in ((f1, 10), (f2, 13), (f3, 19)):
                        q = v - hf
                        if 0 <= q < 12:
                            a = hit_dir[(hf + 3) % 8192]
                            hx += amp * np.cos(a) * np.exp(-q / 2.5); hy += 1.1 * amp * np.sin(a) * np.exp(-q / 2.5)
                            rot += 0.5 * np.sign(np.cos(5 * a)) * np.exp(-q / 3)
                z += min(tz, 0.32)
                rate = 0.14
                for D in drops:                              # the slam lands exactly on the drop frame
                    nxt = [c_ for c_ in vs if c_ > D + 40]
                    rel = min(D + 100, nxt[0] if nxt else D + 100)
                    zd = ease((v - (D - 2)) / 2)
                    z += (0.30 * zd - 0.10 * ease((v - D - 10) / 50)) * (1 - ease((v - rel) / 12))
                    q = v - D
                    if 0 <= q < 12:
                        hx += 18 * np.exp(-q / 2.5); hy -= 22 * np.exp(-q / 2.5); rot += 0.7 * np.exp(-q / 3)
                    if abs(q) < 5:
                        rate = 0.45
                for a, b in breaks:                          # creep in through an 808 break, hand over to the slam
                    if a <= v < b:
                        z += 0.10 * ease((v - a) / max(1, b - a))
                    elif b <= v < b + 8:
                        z += 0.10 * (1 - ease((v - b) / 8))
                z = max(z, 1.02)
                ww, wh = H / z * 9 / 16, H / z
                target = clip.cx[i] - (comp - 0.5) * ww
                trk = target if v == s0 and not (loop and v == N) else trk + (target - trk) * rate
                vx, vy = trk + 0.6 * hx / z, cy * H + (0.5 - FACE_AT) * wh + hy / z
                cr, sr_ = abs(np.cos(np.radians(rot))), abs(np.sin(np.radians(rot)))
                hxe, hye = ww / 2 * cr + wh / 2 * sr_, ww / 2 * sr_ + wh / 2 * cr
                self.vx[v] = np.clip(vx, hxe + 2, W - hxe - 2)
                self.vy[v] = np.clip(vy, hye + 2, H - hye - 2)
                self.z[v], self.r[v] = z, rot
        if loop:                                             # keep the second copy: its frame 0 follows frame N-1
            self.vx, self.vy, self.z, self.r = (a[N:] for a in (self.vx, self.vy, self.z, self.r))
        self.N, self.loop = N, loop

    def matrix(self, i, z=None):
        z = self.z[i] if z is None else z
        s = OH * z / H
        c, sn = np.cos(np.radians(self.r[i])), np.sin(np.radians(self.r[i]))
        vx, vy = self.vx[i], self.vy[i]
        return np.float32([[c / s, sn / s, vx - (c * OW / 2 + sn * OH / 2) / s],
                           [-sn / s, c / s, vy - (-sn * OW / 2 + c * OH / 2) / s]])

    def warp(self, img, i, z=None, interp=cv2.INTER_CUBIC):
        return cv2.warpAffine(img, self.matrix(i, z), (OW, OH), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

    def warp_quarter(self, mask, i):
        M = self.matrix(i).copy(); M[:, 2] /= 4
        return cv2.warpAffine(mask, M, (OW // 4, OH // 4), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def prev(self, i):
        return (i - 1) % self.N if self.loop else i - 1

    def apply(self, img, i, cut):
        out = self.warp(img, i)
        if cut or (i == 0 and not self.loop):
            return out
        p = self.prev(i)
        pz, z = self.z[p], self.z[i]
        if abs(z / pz - 1) > 0.008:                          # fast zoom: radial blur from in-between zooms
            out = sum(self.warp(img, i, pz + (z - pz) * a) for a in (0.25, 0.5, 0.75)) / 4 + out / 4
        s = OH * z / H
        dx, dy = (self.vx[i] - self.vx[p]) * s, (self.vy[i] - self.vy[p]) * s
        if abs(dx) > 5:
            out = cv2.blur(out, (min(121, int(abs(dx) * 0.8)) | 1, 1))
        if abs(dy) > 5:
            out = cv2.blur(out, (1, min(121, int(abs(dy) * 0.8)) | 1))
        return out


# ------------------------------------------------------------------ vertical-frame pieces
YY9, XX9 = np.mgrid[0:OH, 0:OW].astype(np.float32)
VIGN9 = (1 - 0.12 * ((YY9 - OH / 2) / (OH / 2)) ** 2 / 2)[..., None]   # fc.VIGN as it lands in a centre crop


def soft_blur9(img, sigma):
    if sigma < 0.5:
        return img
    small = cv2.resize(img, (OW // 4, OH // 4), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 4), (OW, OH), interpolation=cv2.INTER_LINEAR)


def isoline9(n, level, width_px=2.4):
    ns = cv2.GaussianBlur(n, (0, 0), 3.0)
    gy, gx = np.gradient(ns)
    return np.clip(1 - np.abs(ns - level) / (np.sqrt(gx * gx + gy * gy) * width_px + 1e-6), 0, 1)[..., None]


def transition9(kind, k, L, new, old, n_new):
    """The remix transitions, in the vertical frame (old = last framed picture of the outgoing shot)."""
    u = (k + 1) / L
    if kind in ("sweep", "far_first"):
        v = ease(u)
        p = 1.0 - 1.05 * v if kind == "sweep" else -0.05 + 1.1 * v
        m = (n_new > p) if kind == "sweep" else (n_new < p)
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.5)[..., None]
        out = new * m + old * (1 - m)
        if 0 < v < 1:
            ln = isoline9(n_new, p); out = out * (1 - ln) + fc.WARM_LINE * ln
        return out
    if kind == "whip":
        blur = np.sin(np.pi * u) * 180
        def sb(img, dx):
            o = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]), (OW, OH), borderMode=cv2.BORDER_REFLECT)
            Lb = int(blur) | 1
            return cv2.blur(o, (Lb, 1)) if Lb > 2 else o
        return sb(old, -ease(u / 0.5) * OW * 0.35) if u < 0.5 else sb(new, (1 - ease((u - 0.5) / 0.5)) * OW * 0.35)
    if kind == "depth_dissolve":
        a = np.clip(ease(u) * 1.6 - (1 - n_new) * 0.6, 0, 1).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 3.0)[..., None]
        return new * a + old * (1 - a)
    if kind == "defocus":
        sig = 26 * np.sin(np.pi * u); mix = np.clip((u - 0.35) / 0.3, 0, 1)
        return soft_blur9(old, sig) * (1 - mix) + soft_blur9(new, sig) * mix
    if kind == "scanline":
        y = int(round(ease(u) * (OH + 6))) - 3
        out = old.copy()
        if y > 0:
            out[:min(y, OH)] = new[:min(y, OH)]
        if 0 <= y < OH:
            band = np.zeros((OH, OW), np.float32); band[max(0, y - 1):y + 2] = 1
            glow = cv2.GaussianBlur(band, (0, 0), 7)[..., None]
            out = out * (1 - band[..., None]) + fc.WARM_LINE * band[..., None] + fc.WARM_LINE * 0.25 * glow
        return out
    if kind == "punch":
        s = 1 + 0.08 * (1 - ease(u))
        return cv2.warpAffine(new, cv2.getRotationMatrix2D((OW / 2, OH * 0.45), 0, s), (OW, OH), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(kind)


class LightFX9:
    """Long-exposure trails, bloom and mist in the vertical frame (as fc.LightFX). Resets at cuts."""
    def __init__(self):
        self.trail, self.last = None, -99

    def apply(self, img, lights, vi, cut):
        small = cv2.resize(img, (OW // 4, OH // 4), interpolation=cv2.INTER_AREA)
        hl = np.clip((small.mean(axis=2) / 255 - 0.36) / 0.5, 0, 1) ** 1.3
        if self.trail is None or vi - self.last > 4 or vi <= self.last or cut:
            self.trail = hl.copy()
        else:
            self.trail = cv2.GaussianBlur(np.maximum(self.trail * fc.TRAIL_DECAY ** (vi - self.last), hl), (0, 0), 1.3)
        self.last = vi
        drag = np.clip(self.trail - cv2.GaussianBlur(hl, (0, 0), 1.0), 0, 1)
        add = (0.85 * drag + 0.50 * cv2.GaussianBlur(drag, (0, 0), 5))[..., None] * fc.TRAIL_TINT
        add = add + (0.04 * cv2.GaussianBlur(hl, (0, 0), 10))[..., None] * fc.BLOOM_TINT
        add = cv2.resize(np.clip(add, 0, 255), (OW, OH), interpolation=cv2.INTER_LINEAR)
        return 255 - (255 - np.clip(img, 0, 255)) * (1 - add / 255)


class Flares9(fc.Flares):
    """The remix flares (core, textured anamorphic streak, starburst, veiling glare; no ghost dots), sized for
    the vertical frame."""
    HW, HH = OW // 2, OH // 2

    def render9(self, lights, i, roll, pulse):
        if lights is None or lights.max() < 0.25:
            return None
        nl, lab, st, cen = cv2.connectedComponentsWithStats((lights > 0.25).astype(np.uint8), 8)
        cand = sorted(((st[k, cv2.CC_STAT_AREA] * lights[lab == k].mean(), cen[k][0] * 2, cen[k][1] * 2) for k in range(1, nl)), reverse=True)
        layer = np.zeros((self.HH, self.HW, 3), np.float32)
        pk = 1 + 0.7 * pulse
        for strength, px, py in cand[:4]:
            s = min(1.0, strength / 40) * pk
            if s < 0.2:
                continue
            key = int(round(px / 30)) * 1000 + int(round(py / 30))
            s *= 0.9 + 0.1 * np.random.default_rng(key * 7919 + i).random()
            Ls = 200 * (0.5 + s)
            fall = np.exp(-np.abs(self.xx - px) / Ls) * self._tex(key, self.HW, i)
            y0, y1 = int(max(0, py - 14)), int(min(self.HH, py + 15))
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            for ch, (wid, ln, amp) in enumerate(((1.6, 1.0, 1.0), (1.0, 0.8, 0.75), (0.7, 0.6, 0.55))):
                core = np.exp(-((yy - py) / wid) ** 2) * (fall ** (1 / ln))
                dbl = 0.28 * (np.exp(-((yy - py - 4) / (wid * 0.8)) ** 2) + np.exp(-((yy - py + 4) / (wid * 0.8)) ** 2)) * fall ** 1.6
                layer[y0:y1, :, ch] += (core + dbl) * amp * s * 255 * 0.9
            R = int(80 * (0.5 + s))
            xa, xb, ya, yb = int(max(0, px - R)), int(min(self.HW, px + R)), int(max(0, py - R)), int(min(self.HH, py + R))
            if xb > xa and yb > ya:
                gx, gy = np.meshgrid(np.arange(xa, xb) - px, np.arange(ya, yb) - py)
                rr = np.sqrt(gx ** 2 + gy ** 2) + 1e-3
                th = np.arctan2(gy, gx) - np.radians(roll) + 0.3
                star = np.exp(-rr / (R * 0.35)) * np.maximum(np.cos(3 * th), 0) ** 60
                core = np.exp(-(rr / 2.2) ** 2) + 0.35 * np.exp(-(rr / 14) ** 2) + 0.08 * np.exp(-(rr / 45) ** 2)
                layer[ya:yb, xa:xb] += (core + 0.45 * star)[..., None] * np.array([235, 242, 250], np.float32) * s
            dist = np.hypot(px - self.HW / 2, py - self.HH / 2) / self.HW
            if dist < 0.3 and s > 0.4:
                layer += np.array([200, 215, 230], np.float32) * 0.05 * s * (1 - dist / 0.3)
        return cv2.resize(np.clip(layer, 0, 255), (OW, OH), interpolation=cv2.INTER_LINEAR)


def aberration9(img, a):
    if a < 0.05:
        return img
    out = img.copy()
    for ch, sc in ((2, 1 + 0.007 * a), (0, 1 - 0.007 * a)):
        out[..., ch] = cv2.warpAffine(img[..., ch], cv2.getRotationMatrix2D((OW / 2, OH / 2), 0, sc), (OW, OH), borderMode=cv2.BORDER_REFLECT)
    return out


def finish9(img, i, clip):
    rng = np.random.default_rng(int(clip.key) * 100000 + i)
    grain = cv2.resize(rng.normal(0, 0.014 * 255, (OH // 2, OW // 2)).astype(np.float32), (OW, OH), interpolation=cv2.INTER_LINEAR)
    out = img * VIGN9 + grain[..., None]                    # the remix grain, at footage-pixel size
    if not clip.loop:
        fo = clip.spec.get("fade_out", 8)
        out = out * (1 - ease((i - (clip.N - fo - 1)) / fo))
    return np.clip(out, 0, 255)


# ------------------------------------------------------------------ triptych (vertical: three stacked panels)
def tri_frame(clip, i):
    k, t = clip.seg_t(i)
    opt = clip.edl[k]["opt"]
    gut = 10; ph = (OH - 2 * gut) // 3
    out = np.zeros((OH, OW, 3), np.float32)
    step = opt.get("tri_step", 13)
    for p_, (sh, s_) in enumerate(opt["tri"]):
        if t < p_ * step:                                    # panels land on the half-beats
            continue
        cp, gp, np_, jp = rc.load_f(rc.src_pos(sh, s_, 1.0, t), sh)
        gr = ghost9(gp, np_, jp, sh, clip.pulse(i))
        ch = int(min(H, 760)); cw = int(ch * OW / ph)
        push = 1 + 0.04 * ease(t / 60)
        ch2, cw2 = int(ch / push), int(cw / push)
        x0 = int(np.clip(rc.TRACK[jp] - cw2 / 2, 0, W - cw2)); y0 = int(np.clip(H * 0.46 - ch2 / 2, 0, H - ch2))
        land = ease((t - p_ * step + 1) / 3)                 # each panel snaps in with a quick exposure kick
        pan = cv2.resize(gr[y0:y0 + ch2, x0:x0 + cw2], (OW, ph), interpolation=cv2.INTER_AREA) * (1 + 0.25 * (1 - land))
        out[p_ * (ph + gut):p_ * (ph + gut) + ph] = pan
    return out


# ------------------------------------------------------------------ frame pipeline
def frame_core(clip, i):
    """Framed picture before light/transition: (BGR float OHxOW, depth in frame or None, lights quarter or None, cut)."""
    k, t = clip.seg_t(i)
    opt = clip.edl[k]["opt"]
    cut = (i in clip.cuts) or (i == 0 and not clip.loop)
    if "tri" in opt:
        img = tri_frame(clip, i)
        a = np.radians(clip.cam.r[i] * 0.5)
        M = cv2.getRotationMatrix2D((OW / 2, OH / 2), clip.cam.r[i] * 0.5, 1.02 + 0.5 * (clip.cam.z[i] - 1.1))
        return cv2.warpAffine(img, M, (OW, OH), borderMode=cv2.BORDER_CONSTANT), None, None, cut
    out, n, c, j = compose(clip, i)
    img = clip.cam.apply(out, i, cut)
    lights = clip.cam.warp_quarter(fc.light_sources(c), i)
    tr = opt.get("tr")
    n_out = clip.cam.warp(n, i, interp=cv2.INTER_LINEAR) if tr and t < tr[1] and tr[0] in ("sweep", "far_first", "depth_dissolve") else None
    return img, n_out, lights, cut


def frame(clip, i, fx, vi):
    img, n_out, lights, cut = frame_core(clip, i)
    k, t = clip.seg_t(i)
    tr = clip.edl[k]["opt"].get("tr")
    if tr and t < tr[1] and k > 0:
        s0 = int(clip.starts[k])
        if s0 not in clip.old:
            clip.old[s0] = frame_core(clip, s0 - 1)[0]
        img = transition9(tr[0], t, tr[1], img, clip.old[s0], n_out)
    img = fx["trail"].apply(img, lights, vi, cut)
    fl = fx["flares"].render9(lights, i, clip.cam.r[i], clip.pulse(i))
    if fl is not None:
        img = fc.screen(img, fl)
    img = aberration9(img, clip.impact(i))
    return cv2.cvtColor(finish9(img, i, clip).astype(np.uint8), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ audio
_song = None


def song():
    global _song
    if _song is None:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(WAV), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
        _song = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    return _song


def clip_audio(clip):
    """The clip's audio, exactly N frames long. Loops: 8 bars, the last 10 ms crossfaded into what precedes the
    first downbeat in the song (so the wrap sounds like the song itself), then fitted to N frames by a periodic
    (FFT) resample: 0.1 % faster, 1.8 cents, inaudible, and seamless when it repeats."""
    x = song(); s0 = int(round(clip.a0 * SR)); n_out = int(round(clip.N / FPS * SR))
    if not clip.loop:
        seg = x[s0:s0 + n_out].copy()
        if len(seg) < n_out:
            seg = np.vstack([seg, np.zeros((n_out - len(seg), 2), np.float32)])
        if s0 > 0:
            k = int(0.004 * SR); seg[:k] *= np.linspace(0, 1, k)[:, None]
        fo = int(clip.spec.get("fade_out", 8) / FPS * SR)
        seg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo))[:, None] ** 2
        return seg
    n_in = int(round((clip.b1 - clip.b0) * BAR * SR))
    seg = x[s0:s0 + n_in].copy()
    k = int(0.010 * SR); w = np.linspace(0, 1, k)[:, None]
    seg[-k:] = seg[-k:] * (1 - w) + x[s0 - k:s0] * w
    X = np.fft.rfft(seg.astype(np.float64), axis=0)
    y = np.fft.irfft(X[:n_out // 2 + 1], n=n_out, axis=0) * (n_out / n_in)
    return y.astype(np.float32)


# ------------------------------------------------------------------ render
def build(key):
    clip = Clip(key, CLIPS[key])
    clip.cam = Cam9(clip)
    return clip


def render(clip, out_dir):
    fx = {"trail": LightFX9(), "flares": Flares9()}
    keys = clip.keys()
    aud = out_dir / f".{clip.key}.f32"
    clip_audio(clip).tofile(aud)
    name = out_dir / f"RENDEZVOUS_tiktok_{clip.key}_{clip.name}.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(aud),
                            "-map", "0:v", "-map", "1:a", "-t", f"{clip.N / FPS:.6f}",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-maxrate", "20M", "-bufsize", "40M",
                            "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
                            # loops: the native encoder writes an exact edit list (audio = picture to the sample)
                            "-c:a", "aac" if clip.loop else "aac_at", "-b:a", "320k",
                            "-movflags", "+faststart", str(name)], stdin=subprocess.PIPE)
    if clip.loop:                                            # warm the light trails up on the frames before the seam
        for w in range(clip.N - 10, clip.N):
            frame(clip, w, fx, w - clip.N)
    t0, held = time.time(), None
    for i in range(clip.N):
        if keys[i] or held is None:
            held = np.ascontiguousarray(frame(clip, i, fx, i)).tobytes()
        enc.stdin.write(held)
    enc.stdin.close(); enc.wait(); aud.unlink()
    print(f"{clip.key} {clip.name}: {clip.N} frames ({clip.N / FPS:.2f} s), {keys.sum()} key frames, "
          f"{(time.time() - t0) / 60:.1f} min -> {name.name}", flush=True)
    return name


def preview(clip, frames, out_dir):
    fx = {"trail": LightFX9(), "flares": Flares9()}
    for i in sorted(frames):
        for w in range(max(0, i - 6), i):
            frame(clip, w, fx, w)
        cv2.imwrite(str(out_dir / f"tt{clip.key}_{i:04d}.jpg"), cv2.cvtColor(frame(clip, i, fx, i), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def validate(clip):
    """Every segment stays on its own shot's footage, and colour isolation only runs where the matte is stable."""
    from tiktok_edl import ISO_OK
    problems = []
    for k, e in enumerate(clip.edl):
        frames = [i for i in range(clip.N) if clip.seg_index(i) == k]
        js = [clip.src_pos(i)[0] for i in frames]
        a, b = rc.SHOT[e["shot"]]
        if min(js) < a or max(js) > b - 1 + 1e-6 or ("freeze" not in e["opt"] and max(js) >= b - 1 and len(js) > 2 and js[-1] == js[-2]):
            problems.append(f"seg {k} ({e['shot']}) runs off its shot: {min(js):.0f}-{max(js):.0f} not in {a}-{b - 1}")
        if e["opt"].get("iso"):
            if not any(lo <= min(js) and max(js) <= hi for lo, hi in ISO_OK.get(e["shot"], [])):
                problems.append(f"seg {k} ({e['shot']} {min(js):.0f}-{max(js):.0f}) iso outside the stable matte windows")
    return problems


def describe(clip):
    print(f"{clip.key} {clip.name}: bars {clip.b0}-{clip.b1}, {clip.a0:.2f}-{clip.a1:.2f} s, {clip.N} frames"
          f"{' LOOP r=%.5f' % clip.r if clip.loop else ''} | {len(clip.edl)} segments | punches {[p[0] for p in clip.punch]} | "
          f"drops {clip.drops} | breaks {clip.breaks}")
    for p in validate(clip):
        print("   !!", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--preview")
    ap.add_argument("--describe", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    for key in a.clips.split(","):
        clip = build(key)
        describe(clip)
        if a.describe:
            continue
        if a.preview:
            fr = [int(x) for x in a.preview.split(",")] if a.preview != "auto" else sorted({0, *clip.starts.tolist(), clip.N - 1})
            preview(clip, [f for f in fr if f < clip.N], out)
        else:
            render(clip, out)


if __name__ == "__main__":
    main()
