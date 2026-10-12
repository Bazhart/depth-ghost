#!/usr/bin/env python3
"""
Carousel: four character stills in the Remix depth-ghost look + a last slide that sends people to the GitHub
repo, with a Claude Code mark. Renders 9:16 (TikTok photo mode, 1080x1920) and 4:5 (Instagram, 1080x1350).

usage: python carousel.py OUT_DIR
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
PR = HERE.parent
sys.path.insert(0, str(PR))
import remix_cut as rc  # noqa: E402
import final_cut as fc  # noqa: E402
from final_cut import W, H, ease  # noqa: E402

REPO = "github.com/Bazhart/depth-ghost"
SIZES = {"9x16": (1080, 1920), "4x5": (1080, 1350)}
# (name, shot, source frame, crop centre x as fraction of W or None = tracked singer, zoom)
CHARACTERS = [("01_mic", "A", 20, None, 1.0), ("02_closeup", "F", 1085, "+0.05", 1.0),
              ("03_dancer", "E", 862, None, 1.0), ("04_silhouette", "C", 360, 0.27, 1.0)]
ICE = (222, 234, 246)                                        # RGB, the title's icy fill
NAVY = (30, 52, 82)                                          # RGB, the title's navy outline
CLAUDE = (217, 119, 87)                                      # RGB, Claude orange
FONT = {"black": str(PR / "fonts" / "UnifrakturMaguntia-Book.ttf"), "mono": "/System/Library/Fonts/SFNSMono.ttf",
        "sans": "/System/Library/Fonts/Avenir Next.ttc"}


def ghost_still(j, shot):
    """remix_cut.ghost, except the close-up's focus sits on the face (a still has no time for a focus pull)."""
    c, g, n = rc.load(j)
    if shot == "F":
        sx = int(rc.TRACK[j]); f_face = float(np.median(n[int(H * 0.18):int(H * 0.34), sx - 50:sx + 50]))
        g = fc.focus(g, n, f_face, 9.0)
    out = fc.grade(g, 0.35)
    sx = rc.TRACK[j]
    sd = float(np.median(n[int(H * 0.3):int(H * 0.7), int(max(0, sx - 60)):int(min(W, sx + 60))]))
    band = np.clip(1 - np.abs(n - sd) / 0.22, 0, 1) * (n > 0.12)
    spot = np.exp(-((fc.XX - sx) / (0.11 * W)) ** 2 - ((fc.YY - H * 0.48) / (0.42 * H)) ** 2) * band
    out = out * (1 + 0.32 * cv2.GaussianBlur(spot, (0, 0), 6)[..., None])
    return out, c


def still(shot, j, cxf, zoom, size):
    out, c = ghost_still(j, shot)
    img = fc.LightFX().apply(out, fc.light_sources(c), 0)    # the remix bloom + mist
    img = fc.aberration(img, 0.3)
    img = np.clip(img * fc.VIGN + np.random.default_rng(j).normal(0, 0.014 * 255, (H, W))[..., None], 0, 255)
    ow, oh = size
    ch = int(H / zoom); cw = int(round(ch * ow / oh))
    cx = rc.TRACK[j] if cxf is None else (rc.TRACK[j] + float(cxf) * W if isinstance(cxf, str) else cxf * W)
    x0 = int(np.clip(round(cx - cw / 2), 0, W - cw)); y0 = int(np.clip(round(H * 0.47 - ch / 2), 0, H - ch))
    crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw].astype(np.float32), size, interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def claude_spark(d):
    """The Claude spark in Claude orange: rounded rays of uneven length around a centre (drawn at 4x, smoothed)."""
    S = 4 * d; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    c = S / 2; R = S * 0.48
    lens = [1.0, 0.80, 0.94, 0.76, 0.99, 0.84, 0.91, 0.78, 0.97, 0.82, 0.90, 0.77]
    for k, L in enumerate(lens):
        a = np.radians(-90 + k * 30 + (4 if k % 2 else 0))
        w0, w1 = R * 0.15, R * 0.105
        tip = np.array([c + np.cos(a) * R * L, c + np.sin(a) * R * L])
        base = np.array([c, c]); nrm = np.array([-np.sin(a), np.cos(a)])
        poly = [tuple(base + nrm * w0), tuple(tip + nrm * w1), tuple(tip - nrm * w1), tuple(base - nrm * w0)]
        dr.polygon(poly, fill=CLAUDE + (255,))
        dr.ellipse([tip[0] - w1, tip[1] - w1, tip[0] + w1, tip[1] + w1], fill=CLAUDE + (255,))
    dr.ellipse([c - R * 0.2, c - R * 0.2, c + R * 0.2, c + R * 0.2], fill=CLAUDE + (255,))
    return im.resize((d, d), Image.LANCZOS)


def text_w(draw, s, f, spacing=0):
    return draw.textlength(s, font=f) + spacing * max(0, len(s) - 1)


def spaced(draw, xy, s, f, fill, spacing, anchor="mm"):
    """Letter-spaced text centred on xy."""
    tw = text_w(draw, s, f, spacing); x = xy[0] - tw / 2
    for ch in s:
        draw.text((x, xy[1]), ch, font=f, fill=fill, anchor="lm"); x += draw.textlength(ch, font=f) + spacing


def last_slide(size, lang, bg_rgb):
    ow, oh = size
    k = oh / 1920 if size[1] == 1920 else oh / 1350          # layout scale per format
    tall = size[1] == 1920
    copy = {"fr": ("Fais pareil.", "Suis le lien sur mon GitHub", "fait avec"),
            "en": ("Do the same.", "Follow the link on my GitHub", "made with")}[lang]
    # background: the first still, dimmed and softened, under a dark gradient
    bg = Image.fromarray(bg_rgb).filter(ImageFilter.GaussianBlur(6))
    arr = np.asarray(bg).astype(np.float32) * 0.26
    yy = np.linspace(0, 1, oh)[:, None, None]
    arr = arr * (0.55 + 0.45 * np.sin(np.pi * np.clip(yy * 1.1, 0, 1))) + np.array([7, 9, 12], np.float32) * 0.6
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    cy = {"label": 300, "head": 760, "sub": 975, "pill": 1105, "badge": 1330} if tall else \
         {"label": 170, "head": 500, "sub": 700, "pill": 820, "badge": 1030}
    f_lab = ImageFont.truetype(FONT["sans"], int(30 * (1 if tall else 0.92)), index=2)
    spaced(d, (ow / 2, cy["label"]), "DEPTH  GHOST", f_lab, ICE + (170,), 9)
    d.line([(ow / 2 - 60, cy["label"] + 34), (ow / 2 + 60, cy["label"] + 34)], fill=ICE + (90,), width=2)
    # headline in the RENDEZVOUS blackletter: icy fill, thin navy outline, soft glow
    fh = ImageFont.truetype(FONT["black"], 210 if tall else 180)
    while d.textlength(copy[0], font=fh) > ow * 0.86:
        fh = ImageFont.truetype(FONT["black"], fh.size - 6)
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0)); ImageDraw.Draw(glow).text((ow / 2, cy["head"]), copy[0], font=fh, fill=ICE + (120,), anchor="mm")
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(18)))
    d = ImageDraw.Draw(img)
    d.text((ow / 2, cy["head"]), copy[0], font=fh, fill=ICE + (255,), anchor="mm", stroke_width=max(2, fh.size // 60), stroke_fill=NAVY + (255,))
    f_sub = ImageFont.truetype(FONT["sans"], 50 if tall else 44, index=5)
    d.text((ow / 2, cy["sub"]), copy[1], font=f_sub, fill=ICE + (235,), anchor="mm")
    # the link, in a terminal-style pill
    f_mono = ImageFont.truetype(FONT["mono"], 40 if tall else 36)
    tw = d.textlength(REPO, font=f_mono); ph = 84 if tall else 76; pw = tw + 2 * 46
    x0, y0 = ow / 2 - pw / 2, cy["pill"] - ph / 2
    d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=ph / 2, fill=(10, 12, 16, 210), outline=ICE + (150,), width=2)
    d.text((ow / 2, cy["pill"] + 1), REPO, font=f_mono, fill=ICE + (255,), anchor="mm")
    # made with Claude Code
    f_small = ImageFont.truetype(FONT["sans"], 30 if tall else 27, index=2)
    f_cc = ImageFont.truetype(FONT["sans"], 46 if tall else 41, index=0)
    d.text((ow / 2, cy["badge"] - 62), copy[2], font=f_small, fill=ICE + (150,), anchor="mm")
    sp = claude_spark(64 if tall else 56)
    wcc = d.textlength("Claude Code", font=f_cc); gap = 18
    total = sp.width + gap + wcc; bx = int(ow / 2 - total / 2)
    img.alpha_composite(sp, (bx, int(cy["badge"] - sp.height / 2)))
    d = ImageDraw.Draw(img)
    d.text((bx + sp.width + gap, cy["badge"] + 2), "Claude Code", font=f_cc, fill=(246, 241, 236, 255), anchor="lm")
    return np.asarray(img.convert("RGB"))


def main():
    out = Path(sys.argv[1]).expanduser()
    for fmt, size in SIZES.items():
        d = out / fmt; d.mkdir(parents=True, exist_ok=True)
        first = None
        for k, (name, shot, j, cxf, zoom) in enumerate(CHARACTERS):
            im = still(shot, j, cxf, zoom, size)
            first = im if first is None else first
            Image.fromarray(im).save(d / f"RENDEZVOUS_carousel_{fmt}_{name}.png", optimize=True)
        for lang in ("fr", "en"):
            Image.fromarray(last_slide(size, lang, first)).save(d / f"RENDEZVOUS_carousel_{fmt}_05_github_{lang}.png", optimize=True)
        print(fmt, "done", flush=True)


if __name__ == "__main__":
    main()
