#!/usr/bin/env python3
"""
RENDËZVOUS — Remix Cut for phones: 9:16 and 1:1 from the same edit list as remix_cut.py.

The crop follows the singer through every cut. The layered moments are re-placed for the narrow frames
(9:16: the giant ghost face looms behind the singer and the colour singer stands just left of his ghost;
1:1: both sit in the left part of the square). The triptych becomes three stacked panels (9:16) or three
strips (1:1), landing on the half-beats. No handheld camera on the phone formats, as before.

usage: python remix_social.py OUT_DIR [--preview 100,500,...]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remix_cut as rc  # noqa: E402  (also points final_cut's cut list at the remix cuts)
import final_cut as fc  # noqa: E402
from final_cut import W, H, N, FPS, ease  # noqa: E402

FORMATS = {"9x16": dict(cw=608, size=(1080, 1920)), "1x1": dict(cw=1080, size=(1080, 1080))}
CXS = np.array([rc.compose_track(i) for i in range(N)])


def cfg(fmt, i):
    cx = CXS[i]
    if fmt == "9x16":
        c = {"name": fmt, "loom": (cx, 1.25), "matte": (cx - 0.06 * W, 0.85)}
    else:
        c = {"name": fmt, "loom": (cx - 0.18 * W, 1.12), "matte": (cx - 0.17 * W, 0.95)}
    c["fn"] = lambda j, f=fmt: cfg(f, j)
    return c


def layered(opt):
    return "loom" in opt or "matte" in opt


def needs_format(i):
    """Frames whose composite differs per format: layered segments, and transitions out of them."""
    k, (s0, shot, src0, speed, opt) = rc.seg_of(i)
    if layered(opt):
        return True
    if s0 in fc.TRANSITIONS and i - s0 < fc.TRANSITIONS[s0][1] and k > 0:
        return layered(rc.EDL[k - 1][4])
    return False


def tri_frame(i, fmt, opt, s0):
    t = i - s0
    ow, oh = FORMATS[fmt]["size"]
    gut = 9
    ph = (oh - 2 * gut) // 3
    out = np.zeros((oh, ow, 3), np.float32)
    for p, (sh, s_) in enumerate(opt["tri"]):
        if t < p * 13:                                       # panels land on the half-beats
            continue
        cp, gp, np_, jp = rc.load_f(rc.src_pos(sh, s_, 1.0, t), sh)
        gr = rc.ghost(gp, np_, jp, sh, i)
        if fmt == "9x16":                                    # stacked panels, cropped around each singer
            ch = 700; cw = int(ch * ow / ph)
            x0 = int(np.clip(rc.TRACK[jp] - cw / 2, 0, W - cw)); y0 = int(np.clip(H * 0.45 - ch / 2, 0, H - ch))
        else:                                                # full-width strips
            cw = W; ch = int(W * ph / ow)
            x0 = 0; y0 = int(np.clip(H * 0.42 - ch / 2, 0, H - ch))
        out[p * (ph + gut):p * (ph + gut) + ph] = cv2.resize(gr[y0:y0 + ch, x0:x0 + cw], (ow, ph), interpolation=cv2.INTER_AREA)
    return out


def finish_any(img, i):
    rng = np.random.default_rng(i)
    out = img + rng.normal(0, 0.014 * 255, img.shape[:2])[..., None]
    return np.clip(out * (1 - ease((i - (N - 11)) / 10)), 0, 255)


def drop_zoom(i):
    """Phone formats: crop zoom that lands on the 808 at the drop (13.70 s), holds, eases back by 17 s."""
    zd = ease((i - (fc.DROP_F - 2)) / 2)
    return 1 + (0.40 * zd - 0.15 * ease((i - fc.DROP_F - 10) / 60)) * (1 - ease((i - 500) / 12))


def frames_at(i, fx, flares):
    """Both phone formats for output frame i, as RGB uint8."""
    k, (s0, shot, src0, speed, opt) = rc.seg_of(i)
    res = {}
    if "tri" in opt:
        for f in FORMATS:
            res[f] = cv2.cvtColor(finish_any(tri_frame(i, f, opt, s0), i).astype(np.uint8), cv2.COLOR_BGR2RGB)
        return res
    per_fmt = needs_format(i)
    shared = None if per_fmt else rc.compose(i)
    for f, spec in FORMATS.items():
        out, n, c, _ = rc.compose(i, cfg(f, i)) if per_fmt else shared
        lights = fc.light_sources(c)
        img = fx[f].apply(out, lights, i)
        fl = flares.render(lights, CXS[i], H / 2, i)
        if fl is not None:
            img = fc.screen(img, fl)
        img = fc.finish(fc.aberration(img, fc.impact(i)), i)
        zc = drop_zoom(i)
        cw, size = int(round(spec["cw"] / zc)), spec["size"]
        ch = int(round(H / zc))
        x0 = int(np.clip(round(CXS[i] - cw / 2), 0, W - cw))
        y0 = int(np.clip(round(H * 0.46 - ch / 2), 0, H - ch))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        if crop.shape[1] != size[0] or crop.shape[0] != size[1]:
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4)
        res[f] = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preview")
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    fx = {f: fc.LightFX() for f in FORMATS}
    flares = fc.Flares()
    if a.preview:
        for i in sorted(int(x) for x in a.preview.split(",")):
            for w in range(max(0, i - 8), i):
                frames_at(w, fx, flares)
            for f, img in frames_at(i, fx, flares).items():
                cv2.imwrite(str(out / f"remix_{f}_{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("previews done"); return
    keys = fc.poster_keys()
    enc = {}
    for f, spec in FORMATS.items():
        w, h = spec["size"]
        enc[f] = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                   "-r", str(FPS), "-i", "-", "-i", fc.AUDIO, "-map", "0:v", "-map", "1:a", "-t", f"{N / FPS:.6f}",
                                   "-af", f"afade=t=out:st={N / FPS - 0.35:.3f}:d=0.35", "-c:v", "libx264", "-preset", "slow",
                                   "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac_at", "-b:a", "320k", "-movflags", "+faststart",
                                   str(out / f"RENDEZVOUS_remix_{f}_HQ.mp4")], stdin=subprocess.PIPE)
    print(f"phone remix: {keys.sum()} key frames (posterize time)", flush=True)
    t0, held = time.time(), None
    for i in range(N):
        if keys[i] or held is None:
            held = {f: np.ascontiguousarray(img).tobytes() for f, img in frames_at(i, fx, flares).items()}
        for f, buf in held.items():
            enc[f].stdin.write(buf)
        if (i + 1) % 200 == 0 or i + 1 == N:
            el = time.time() - t0
            print(f"frame {i + 1}/{N}  ~{el / (i + 1) * (N - i - 1) / 60:.1f} min left", flush=True)
    for e in enc.values():
        e.stdin.close(); e.wait()
    print("done", flush=True)


if __name__ == "__main__":
    main()
