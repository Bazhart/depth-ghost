#!/usr/bin/env python3
"""
RENDËZVOUS — Remix Cut. A re-edit of the same 47 s on the same music: the original order and section
boundaries (every original cut stays on its beat, with its transition), but long holds are broken up with
cutaways on the beat, plus layering: a depth-masked double exposure ("loom"), a colour matte of the real
singer inside the ghost room, a triptych, and a strobe of 4-frame cuts through the second 808 break.

16:9 only. Reuses the Scan Cut look and effects from final_cut.py (grade, light trails, flares, colour
punches, rings, glitch, posterize time). Footage is read at random from frames/ (extracted JPEGs).

usage: python remix_cut.py OUT.mp4 [--preview 100,500,...]
"""
import argparse
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_cut as fc  # noqa: E402
from final_cut import W, H, N, FPS, BAR, FREEZE, ease  # noqa: E402

SHOT = {"A": (0, 305), "B": (305, 319), "C": (319, 538), "D": (538, 715),
        "E": (715, 919), "F": (919, 1151), "G": (1151, 1318), "H": (1318, 1407)}
ENERGY = {"A": 0.85, "B": 1.0, "C": 0.5, "D": 0.75, "E": 1.25, "F": 0.6, "G": 0.9, "H": 1.25}
COMP = {"A": 0.60, "B": None, "C": 0.40, "D": 0.58, "E": 0.60, "F": 0.52, "G": 0.62, "H": 0.62}

# Edit list: (output start frame, source shot, source start frame, speed, options). Each segment runs to the
# next one's start. Options: loom=(shot, src, speed, opacity) depth-masked double exposure; matte=(shot, src,
# speed) the real singer in colour cut out by depth; tri=[(shot, src) x3] triptych; bleed=True colour bleed.
EDL = [
    (0, "A", 0, 1.0, {}),                                   # opening scan, artist at the mic
    (101, "C", 352, 1.0, {}),                               # downbeat: whip-pan angle
    (154, "A", 154, 1.0, {}),                               # triple 5.1 s, hit 1
    (166, "E", 790, 1.0, {}),                               # hit 2: the dancer
    (179, "A", 179, 0.30, {}),                              # hit 3 crash: slow-mo on the arm sweep
    (230, "G", 1160, 1.0, {}),                              # second wide angle
    (257, "A", 250, 1.0, {"loom": ("F", 1000, 1.0, 0.7)}),  # the close-up face looms over the room
    (305, "B", 305, 1.0, {}),                               # stairs flash (far-first reveal)
    (319, "C", 319, 1.0, {}),                               # untouched framing until the break (whip in)
    (387, "C", 387, 2.5, {}),                               # 808 break: rush
    (411, "C", 447, 0.25, {}),                              # THE DROP: slow-mo
    (451, "H", 1340, 1.0, {}),                              # hoodie handshake
    (475, "C", 470, 1.0, {}),
    (512, "E", 860, 1.0, {}),                               # downbeat: dancer
    (538, "D", 538, 1.0, {}),                               # stairs (depth sweep in)
    (604, "D", 604, 1.0, {}),                               # triple 20.1 s, hit 1
    (610, "A", 210, 1.0, {}),                               # hit 2
    (616, "D", 616, 0.25, {}),                              # hit 3 crash slow-mo
    (660, "D", 640, 1.0, {"loom": ("F", 1086, 1.0, 0.65)}),  # face looms over the stairs
    (715, "E", 715, 1.0, {}),                               # dancer (depth dissolve in)
    (770, "G", 1230, 1.0, {}),
    (809, "E", 809, 1.0, {}),                               # triple 27.0 s, hit 1
    (816, "C", 500, 1.0, {}),                               # hit 2
    (822, "E", 822, 0.25, {}),                              # hit 3 crash slow-mo
    (870, "E", 860, 1.0, {"tri": [("E", 860), ("F", 960), ("D", 650)]}),   # triptych lands on the half-beats
    (919, "F", 919, 1.0, {}),                               # close-up (defocus in)
    (944, "H", 1380, 1.0, {}),                              # double triple: hit 2
    (951, "F", 951, 1.0, {}),                               # hit 3 crash
    (957, "E", 900, 1.0, {}),                               # second triple: hit 1
    (964, "C", 430, 1.0, {}),                               # hit 2
    (970, "F", 970, 1.0, {}),                               # hit 3 crash, focus pull lands
    (1009, "F", 1009, 0.35, {"bleed": True}),               # colour bleed, slow-mo (half as long as before)
    (1047, "G", 1190, 1.0, {"matte": ("F", 1030, 0.5)}),    # the real singer in colour inside the ghost room
    (1086, "D", 670, 1.0, {}),
    (1128, "F", 1128, 1.0, {}),                             # triple 37.7 s, hit 1
    (1136, "E", 845, 1.0, {}),                              # hit 2
    (1141, "F", 1141, 1.0, {}),                             # hit 3 crash
    (1151, "G", 1151, 1.0, {}),                             # wide (scan-line wipe in)
    (1210, "G", 1210, 1.0, {}),                             # 808 break: strobe of 8-frame cuts (kept under the flash limit)
    (1218, "C", 330, 1.0, {}),
    (1226, "F", 990, 1.0, {}),
    (1234, "G", 1234, 0.25, {}),                            # the 808 slams back: slow-mo crash
    (1274, "E", 760, 1.0, {}),
    (1318, "H", 1318, 1.0, {}),                             # hoodie (punch-in), freeze, colour returns
]
STARTS = [e[0] for e in EDL]
CUTS = STARTS[1:]
fc.CUT_F = CUTS                                             # light trails, camera blur and posterize follow these cuts
TRACK = fc.track_singer()                                   # singer position per source frame


def seg_of(i):
    k = int(np.searchsorted(STARTS, i, side="right") - 1)
    return k, EDL[k]


def src_pos(shot, s0, speed, t):
    a, b = SHOT[shot]
    return float(np.clip(s0 + t * speed, a, b - 1))


@lru_cache(maxsize=24)
def load(j):
    c = cv2.imread(str(HERE / "frames" / "colour" / f"{j:06d}.jpg")).astype(np.float32)
    g = cv2.cvtColor(cv2.imread(str(HERE / "frames" / "ghost" / f"{j:06d}.jpg")), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return c, g, fc.depth(j)


def load_f(p, shot):
    """Footage at a fractional source position: blend neighbours in slow-mo (never across shots)."""
    j = int(np.floor(p)); fr = p - j
    c, g, n = load(j)
    if fr > 0.05 and j + 1 < SHOT[shot][1]:
        c1, g1, n1 = load(j + 1)
        c, g, n = c * (1 - fr) + c1 * fr, g * (1 - fr) + g1 * fr, n * (1 - fr) + n1 * fr
    return c, g, n, j


def ghost(g, n, j, shot, i, keylight=True):
    if shot == "F":                                         # close-up: focus pull mic -> face, on its own time
        t = (j - SHOT["F"][0]) / FPS
        g = fc.focus(g, n, 0.92 - 0.30 * ease((t - 1.5) / 1.2), 9.0)
    out = fc.grade(g, fc.pulse(i))
    if keylight:
        sx = TRACK[j]
        sd = float(np.median(n[int(H * 0.3):int(H * 0.7), int(max(0, sx - 60)):int(min(W, sx + 60))]))
        band = np.clip(1 - np.abs(n - sd) / 0.22, 0, 1) * (n > 0.12)
        spot = np.exp(-((fc.XX - sx) / (0.11 * W)) ** 2 - ((fc.YY - H * 0.48) / (0.42 * H)) ** 2) * band
        out = out * (1 + 0.32 * cv2.GaussianBlur(spot, (0, 0), 6)[..., None])
    return out


def place(imgs, src_x, dst_x, scale):
    """Move a layer so its subject (at src_x) stands at dst_x, scaled about head height."""
    M = np.float32([[scale, 0, dst_x - scale * src_x], [0, scale, H * 0.45 * (1 - scale)]])
    return [cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for im in imgs]


def singer_band(n, x):
    """The close-up singer only, head included: his depth band without the microphone in front of him or the
    room behind, kept to the one connected shape around his body."""
    band = (np.clip((n - 0.30) / 0.06, 0, 1) * np.clip((0.86 - n) / 0.05, 0, 1)).astype(np.float32)
    small = cv2.resize((band > 0.5).astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    nl, lab = cv2.connectedComponents(small, connectivity=8)
    px, py = int(np.clip(x / 4, 0, W // 4 - 1)), int(H * 0.65 / 4)
    k = lab[py, px]
    if k == 0:                                               # torso point missed: take the largest shape
        counts = np.bincount(lab.ravel())[1:]
        k = int(np.argmax(counts)) + 1 if len(counts) else 0
    keep = cv2.resize((lab == k).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) if k else np.zeros((H, W), np.float32)
    return band * keep


def pop(c):
    grey = c.mean(axis=2, keepdims=True)
    return np.clip((grey + 1.25 * (c - grey)) * 1.15, 0, 255)


_old = {}


MASTER_CFG = {"name": "master", "loom": (0.28 * W, 1.12), "matte": (0.32 * W, 1.0)}


def compose(i, cfg=None):
    """Everything before the camera for output frame i. Returns (frame, depth, colour, track x).
    cfg places the layered subjects (loom / matte) for the output format."""
    cfg = cfg or MASTER_CFG
    k, (s0, shot, src0, speed, opt) = seg_of(i)
    t = i - s0
    if i >= FREEZE:                                         # ending: hold the freeze frame
        t = FREEZE - s0
    c, g, n, j = load_f(src_pos(shot, src0, speed, t), shot)

    if "tri" in opt:                                        # triptych: three panels land on the half-beats
        out = np.zeros((H, W, 3), np.float32)
        pw, gut = 680, 9
        for p_, (sh, s_) in enumerate(opt["tri"]):
            if t < p_ * 13:
                continue
            cp, gp, np_, jp = load_f(src_pos(sh, s_, 1.0, t), sh)
            gr = ghost(gp, np_, jp, sh, i)
            x0 = int(np.clip(TRACK[jp] - pw / 2, 0, W - pw))
            out[:, p_ * (pw + gut):p_ * (pw + gut) + pw] = gr[:, x0:x0 + pw]
        return out, n, c, W / 2

    out = ghost(g, n, j, shot, i)
    cg = fc.colour(c)

    if i < 64:                                              # opening: scan from colour into ghost
        p = 1.05 - 1.15 * ease((i - 8) / 44)
        m = cv2.GaussianBlur((n > p).astype(np.float32), (0, 0), 2.0)[..., None]
        out = out * m + cg * (1 - m)
        if 0 < ease((i - 8) / 44) < 1:
            ln = fc.isoline(n, p); out = out * (1 - ln) + fc.WARM_LINE * ln

    if opt.get("bleed"):                                    # colour bleeds into the singer, then recedes
        L = STARTS[k + 1] - s0
        a = ease((t + 1) / 6) * (1 - ease((t - (L - 7)) / 6))
        band = np.clip(1 - np.abs(n - 0.62) / 0.14, 0, 1) ** 0.7 * (n > 0.40)
        m = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.5)[..., None] * a
        out = out * (1 - m) + pop(c) * m

    if "loom" in opt:                                       # double exposure: another shot's subject, depth-masked
        sh, s_, sp, op = opt["loom"]
        lc, lg, ln_, lj = load_f(src_pos(sh, s_, sp, t), sh)
        lgr = ghost(lg, ln_, lj, sh, i, keylight=False)
        m = cv2.GaussianBlur(singer_band(ln_, TRACK[lj]), (0, 0), 6)
        lgr, m = place([lgr, m], TRACK[lj], *cfg["loom"])         # master: a giant face in the empty left third of the room
        L = STARTS[k + 1] - s0
        fade = ease((t + 1) / 8) * (1 - ease((t - (L - 6)) / 6))
        out = fc.screen(out, lgr * m[..., None] * op * fade)

    if "matte" in opt:                                      # the real singer, in colour, inside the ghost room
        sh, s_, sp = opt["matte"]
        mc, mg, mn, mj = load_f(src_pos(sh, s_, sp, t), sh)
        mpop, mb = place([pop(mc), cv2.GaussianBlur(singer_band(mn, TRACK[mj]), (0, 0), 3)], TRACK[mj], *cfg["matte"])
        m = mb[..., None]
        L = STARTS[k + 1] - s0
        m = m * ease((t + 1) / 5) * (1 - ease((t - (L - 5)) / 5))
        edge = np.clip(cv2.GaussianBlur(m[..., 0], (0, 0), 14) - m[..., 0], 0, 1)[..., None]
        out = out * (1 - m) + mpop * m + np.array([235, 242, 250], np.float32) * 0.35 * edge

    for hf in fc.RING_HITS:                                 # depth-scan rings on 808 + open-hat hits
        q = i - hf
        if 0 <= q < 10:
            ln = fc.isoline(n, 1.0 - 1.1 * ease((q + 1) / 10), 2.0) * (0.7 * (1 - q / 10) ** 1.2)
            out = out * (1 - ln) + np.array([255, 236, 205], np.float32) * ln
    pa = fc.punch_alpha(i, n)                               # colour punches on the impacts
    if np.max(pa) > 0:
        out = out * (1 - pa) + fc.punchy(c) * pa
    ex = fc.impact(i)
    if ex > 0.05:
        out = out * (1 + 0.16 * min(ex, 1.2))
    for gs, gl in fc.GLITCH:
        if gs <= i < gs + gl:
            out = fc.glitch(out, i, i - gs)

    if i >= FREEZE:                                         # colour returns from the back of the room forwards
        v = ease((i - FREEZE) / 26); p = -0.05 + 1.1 * v
        m = cv2.GaussianBlur((n < p).astype(np.float32), (0, 0), 2.0)[..., None]
        out = out * (1 - m) + cg * m
        if 0 < v < 1:
            ln = fc.isoline(n, p); out = out * (1 - ln) + fc.WARM_LINE * ln

    if s0 in fc.TRANSITIONS:                                # the original transitions at the original cuts
        kind, L = fc.TRANSITIONS[s0]
        if t < L:
            key = (s0, cfg["name"])
            if key not in _old:
                _old[key] = compose(s0 - 1, cfg_at(s0 - 1, cfg))[0]
            out = fc.transition(kind, t, L, out, _old[key], n)
    return out, n, c, TRACK[j]


def cfg_at(i, cfg):
    """Re-evaluate a format's layer placement for another frame (placements can follow the crop)."""
    return cfg if cfg["name"] == "master" or "fn" not in cfg else cfg["fn"](i)


class Camera2(fc.Camera):
    """Handheld operator per segment, keyed to the music (pulses, triples, drops), tracking the singer."""
    def __init__(self, cxs):
        self.vx, self.vy, self.z, self.r = (np.zeros(N) for _ in range(4))
        hit_dir = np.random.default_rng(7).uniform(0, 2 * np.pi, 4096)
        trk = 0.0
        for k, (s0, shot, src0, speed, opt) in enumerate(EDL):
            s1 = STARTS[k + 1] if k + 1 < len(EDL) else N
            rng = np.random.default_rng(300 + k)
            f = rng.uniform(0.18, 1.3, (3, 4)); ph = rng.uniform(0, 2 * np.pi, (3, 4)); am = rng.uniform(0.5, 1, (3, 4))
            fj = rng.uniform(3.5, 6.0, 2); pj = rng.uniform(0, 2 * np.pi, 2)
            tri = "tri" in opt
            e0 = 0.3 if tri else ENERGY[shot]
            drift = rng.choice([-1, 1]) * 0.03
            for i in range(s0, s1):
                t = (i - s0) / FPS
                u = (i - s0) / max(1, s1 - s0 - 1)
                e = e0 * (0.35 if i >= FREEZE else 1.0)
                n3 = [(am[c] * np.sin(2 * np.pi * f[c] * t + ph[c])).sum() / am[c].sum() for c in range(3)]
                hx = e * (21 * n3[0] + 2.6 * np.sin(2 * np.pi * fj[0] * t + pj[0]))
                hy = e * (14 * n3[1] + 2.0 * np.sin(2 * np.pi * fj[1] * t + pj[1]))
                rot = e * 0.65 * n3[2]
                past = fc.BEAT_F[fc.BEAT_F <= i]
                if len(past):
                    hy += e * 4 * np.exp(-(i - past[-1]) / 4)
                z = (1.04 if tri else 1.10) + (drift * ease(u) if drift > 0 else -drift * (1 - ease(u)))   # slow push in or pull out
                pk = 0.5 if tri else 1.1
                if i < FREEZE:
                    for hf in fc.JOINT_F:
                        q = i - hf
                        if 0 <= q < 12 and hf not in fc.TRIP_HITS:
                            env = fc._pulse_env(q); a = hit_dir[hf % 4096]
                            z += 0.055 * pk * env
                            hx += 14 * pk * np.cos(a) * np.exp(-q / 2.5); hy += 10 * pk * np.sin(a) * np.exp(-q / 2.5)
                            rot += 0.35 * pk * np.sign(np.sin(3 * a)) * np.exp(-q / 3)
                    tz = 0.0
                    for g in fc.TRIP_F:
                        f1, f2, f3 = g
                        v = (0.07 * ease((i - f1 + 1) / 2) + 0.07 * ease((i - f2 + 1) / 2)
                             + 0.16 * ease((i - f3 + 1) / 2)) * (1 - ease((i - (f3 + 9)) / 12))
                        tz += v * (0.8 if shot == "F" else 1.0) * (0.4 if tri else 1.0)
                        for hf, amp in ((f1, 16), (f2, 20), (f3, 30)):
                            q = i - hf
                            if 0 <= q < 12:
                                a = hit_dir[(hf + 3) % 4096]
                                hx += amp * np.cos(a) * np.exp(-q / 2.5); hy += 0.7 * amp * np.sin(a) * np.exp(-q / 2.5)
                    z += min(tz, 0.34)
                    zd = ease((i - (fc.DROP_F - 2)) / 2)             # THE DROP: the zoom lands exactly on the 808 frame,
                    z += (0.30 * zd - 0.12 * ease((i - fc.DROP_F - 10) / 60)) * (1 - ease((i - 500) / 12))   # holds, eases off by 17 s
                    zw = ease((i - (fc.WHIP_F - 2)) / 2)             # second slam: same landing, shorter hold
                    z += 0.20 * zw * (1 - ease((i - fc.WHIP_F - 10) / 16))
                    for df in (fc.DROP_F, fc.WHIP_F):                # impact jolt on both slams
                        q = i - df
                        if 0 <= q < 12:
                            hx += 30 * np.exp(-q / 2.5); hy -= 18 * np.exp(-q / 2.5); rot += 0.6 * np.exp(-q / 3)
                comp = None if tri else (0.47 if s0 == fc.DROP_F else COMP[shot])
                target = 0.0 if comp is None else 0.75 * (cxs[i] - comp * W)
                rate = 0.45 if abs(i - fc.DROP_F) < 5 else 0.12  # the drop: whip onto the singer
                trk = target if (i == s0 and s0 != fc.DROP_F) else trk + (target - trk) * rate
                vx, vy = W / 2 + trk + hx / z, H / 2 + hy / z
                m = 6 + abs(np.sin(np.radians(rot))) * W / 2
                hw, hh = W / (2 * z), fc._visible_h(i) / (2 * z)
                self.vx[i] = np.clip(vx, hw + m, W - hw - m)
                self.vy[i] = np.clip(vy, hh + m, H - hh - m)
                self.z[i], self.r[i] = z, rot


def hud(img, i):
    a = ease((i - 166) / 15) * (1 - ease((i - 1146) / 10))
    if a <= 0.003:
        return img
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(fc.MONO[0], 14)
    pil = Image.fromarray(img); layer = Image.new("RGBA", pil.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    d.text((W - 40, H - BAR / 2), f"TC {fc.tc(i)}", font=f, fill=(244, 234, 220, int(255 * 0.42 * a)), anchor="rm")
    return np.array(Image.alpha_composite(pil.convert("RGBA"), layer).convert("RGB"))


def frame(i, cam, fx, cxs):
    out, n, c, _ = compose(i)
    framed = cam.apply(out, i)
    lights = fc.light_sources(c) if "tri" not in seg_of(i)[1][4] else None
    lq = cam.warp_quarter(lights, i) if lights is not None else None
    framed = fx["trail"].apply(framed, lq, i)
    fl = fx["flares"].render(lq, W / 2, H / 2, i, cam.r[i])
    if fl is not None:
        framed = fc.screen(framed, fl)
    framed = fc.aberration(framed, fc.impact(i))
    F = fc.finish(framed, i)
    b = int(round(BAR * ease(i / 24)))
    if b:
        F[:b] = (14, 10, 8); F[H - b:] = (14, 10, 8)
    return hud(cv2.cvtColor(F.astype(np.uint8), cv2.COLOR_BGR2RGB), i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preview")
    a = ap.parse_args()
    cxs = np.array([compose_track(i) for i in range(N)])
    cam = Camera2(cxs)
    fx = {"trail": fc.LightFX(), "flares": fc.Flares()}
    out = Path(a.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    if a.preview:
        for i in sorted(int(x) for x in a.preview.split(",")):
            for w in range(max(0, i - 8), i):                # warm up the light trails
                frame(w, cam, fx, cxs)
            cv2.imwrite(str(out.parent / f"remix_preview_{i:04d}.jpg"), cv2.cvtColor(frame(i, cam, fx, cxs), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("previews done"); return
    keys = fc.poster_keys()
    print(f"{len(EDL)} segments, {len(CUTS)} cuts; posterize time: {keys.sum()} key frames (~{keys.sum() / N * FPS:.1f} fps)", flush=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-i", fc.AUDIO, "-map", "0:v", "-map", "1:a", "-t", f"{N / FPS:.6f}",
                            "-af", f"afade=t=out:st={N / FPS - 0.35:.3f}:d=0.35", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac_at", "-b:a", "320k", "-movflags", "+faststart",
                            str(out)], stdin=subprocess.PIPE)
    t0, held = time.time(), None
    for i in range(N):
        if keys[i] or held is None:
            held = np.ascontiguousarray(frame(i, cam, fx, cxs)).tobytes()
        enc.stdin.write(held)
        if (i + 1) % 200 == 0 or i + 1 == N:
            el = time.time() - t0
            print(f"frame {i + 1}/{N}  ~{el / (i + 1) * (N - i - 1) / 60:.1f} min left", flush=True)
    enc.stdin.close(); enc.wait()
    print("done", flush=True)


def compose_track(i):
    k, (s0, shot, src0, speed, opt) = seg_of(i)
    t = min(i, FREEZE) - s0
    return W / 2 if "tri" in opt else TRACK[int(src_pos(shot, src0, speed, t))]


if __name__ == "__main__":
    main()
