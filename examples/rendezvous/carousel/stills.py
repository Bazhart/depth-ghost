#!/usr/bin/env python3
"""Hero stills in the Remix depth-ghost look (remix_cut.ghost, unchanged) for a carousel: portrait crops of
the characters, with the remix's light (mist / bloom, flares from the real lights), aberration, grain, vignette."""
import sys
from pathlib import Path
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent; PR = HERE.parent
sys.path.insert(0, str(PR)); sys.path.insert(0, str(PR / "tiktok"))
import remix_cut as rc, final_cut as fc
from final_cut import W, H


def still(j, shot, cx, size=(1080, 1350), cy=0.47, zoom=1.0, pulse=0.35, flare=True):
    c, g, n = rc.load(j)
    fc.pulse = lambda i: pulse
    out = rc.ghost(g, n, j, shot, 0)
    ow, oh = size
    ch = int(H / zoom); cw = int(round(ch * ow / oh))
    x0 = int(np.clip(round(cx - cw / 2), 0, W - cw)); y0 = int(np.clip(round(cy * H - ch / 2), 0, H - ch))
    lights = fc.light_sources(c)
    fx = fc.LightFX(); img = fx.apply(out, lights, 0)        # bloom + mist (one frame: no trail yet)
    if flare:
        fl = fc.Flares().render(lights, x0 + cw / 2, H / 2, 0)
        if fl is not None:
            img = fc.screen(img, fl)
    img = fc.aberration(img, 0.35)
    rng = np.random.default_rng(j)
    img = np.clip(img * fc.VIGN + rng.normal(0, 0.014 * 255, (H, W))[..., None], 0, 255)
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop.astype(np.float32), size, interpolation=cv2.INTER_LANCZOS4).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    cands = [("A", 20), ("A", 60), ("A", 190), ("A", 265), ("G", 1200), ("G", 1262), ("G", 1300),
             ("F", 1030), ("F", 1085), ("F", 1125), ("E", 870), ("E", 895), ("E", 915),
             ("E", 750, 0.44), ("E", 790, 0.44), ("H", 1335), ("H", 1360), ("H", 1395), ("D", 700), ("C", 360, 0.27)]
    tiles = []
    for cnd in cands:
        shot, j = cnd[:2]; cx = cnd[2] * W if len(cnd) > 2 else rc.TRACK[j]
        im = cv2.resize(still(j, shot, cx, (1080, 1350)), (216, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{shot}{j}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(im)
    rows = [np.hstack(tiles[k:k + 10]) for k in range(0, 20, 10)]
    cv2.imwrite(str(HERE / "qa" / "candidates.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("ok")
