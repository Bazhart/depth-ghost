#!/usr/bin/env python3
"""
depth_ghost.py — turn a regular photo (or video) into the "ghost depth" look:
near things glow white, far things fall off into black.

How it works
  1. A monocular depth model (Depth Anything V2) guesses how far every pixel is.
  2. The raw depth is normalised with percentiles, so a few outliers don't wash it out.
  3. A gamma curve + black point crush the background into darkness (the "ghost" falloff).
  4. A self-guided filter smooths surfaces into that soft, plastic look while keeping
     silhouettes crisp.
  5. Optional extras: --detail (folds/layers), --lift (bring far buildings out of the
     black), --height (render small photos at full resolution), --pillarbox (16:9 frame).

Usage
  python depth_ghost.py photo.jpg                       # -> photo_depth.png
  python depth_ghost.py photo.jpg --gamma 2.2 --pillarbox
  python depth_ghost.py clip.mp4 --model small          # video, audio kept
  python depth_ghost.py folder/                         # every image in a folder
  python depth_ghost.py IMG_1234.HEIC --gamma 2.5 --exposure 1.7 --detail 1.5 --lift 0.45
"""

import os

# Let ops that Apple's MPS backend lacks (bicubic resize inside the model) fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import platform
import time
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:  # iPhone .HEIC photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",  # fast, fine for video
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",  # best edges, slowest
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
BAR_COLOR = (11, 13, 13)  # the near-black bars around the reference frame


# ---------------------------------------------------------------- model

class DepthModel:
    def __init__(self, size: str, input_res: int, device: str = "auto"):
        name = MODELS[size]
        if device != "auto":
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and platform.machine() == "arm64":
            self.device = "mps"
        else:
            # Intel Macs report MPS (AMD GPUs) but it's slower than the CPU there.
            self.device = "cpu"
        print(f"loading {name} on {self.device} ...", file=sys.stderr)
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()
        # Bigger input = finer edges (fingers, railings) but slower. Must be a multiple of 14.
        self.input_res = max(14, input_res // 14 * 14)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb uint8 HxWx3 -> float32 HxW relative inverse depth (bigger = closer)."""
        h, w = rgb.shape[:2]
        inputs = self.processor(
            images=Image.fromarray(rgb),
            return_tensors="pt",
            size={"height": self.input_res, "width": self.input_res},
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            do_resize=True,
        ).to(self.device)
        try:
            pred = self.model(**inputs).predicted_depth.float().cpu()  # 1 x h' x w'
        except RuntimeError as e:
            if self.device == "cpu":
                raise
            # Big/portrait inputs can exceed GPU memory: finish on CPU instead.
            print(f"GPU out of memory ({e}); falling back to CPU", file=sys.stderr)
            self.device = "cpu"
            self.model = self.model.to("cpu")
            pred = self.model(**inputs.to("cpu")).predicted_depth.float()
        # Upsample on CPU: bicubic isn't implemented on Apple's MPS backend.
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        )
        return pred.squeeze().float().cpu().numpy()


# ---------------------------------------------------------------- look

def percentile_range(depth: np.ndarray, lo_pct: float, hi_pct: float):
    return np.percentile(depth, lo_pct), np.percentile(depth, hi_pct)


def tone_curve(args, samples=2048):
    """Depth (0 far .. 1 near) -> brightness, as a lookup table.

    Built as a 1-D curve so it can be forced to be monotonic: brightness must
    only ever rise with nearness. A dip anywhere in the curve shows up as a dark
    outline around every silhouette (edges pass through all in-between depths).
    """
    n = np.linspace(0, 1, samples)
    # Black point: everything past this distance is pure black.
    d = np.clip((n - args.black) / max(1 - args.black, 1e-6), 0, 1)
    # Gamma > 1 pushes mid/far tones down -> the background melts into shadow
    # while the nearest subject stays bright.
    d = d ** args.gamma
    # Exposure on the near end, with a soft shoulder instead of a hard clip:
    # bright subjects glow toward white but keep their folds/planes readable.
    d = d * args.exposure
    k = 0.8
    d = np.where(d < k, d, k + (1 - k) * (1 - np.exp(-(d - k) / (1 - k))))

    # Background lift: the model squeezes everything far away into a sliver near 0,
    # so distant shapes (buildings, skylines) vanish. Log-stretch that far band into
    # faint greys (sky stays black), then ease into the main curve without dipping.
    if args.lift > 0:
        band = args.lift_band
        far = np.log1p(20 * np.clip(n / band, 0, 1)) / np.log1p(20) * args.lift
        p = int(np.argmax(n >= band))                       # end of the lifted band
        # Bridge to a point clearly above the lift level, so mid-distances keep a
        # rising gradient (a flat plateau would make them one washed-out grey).
        catch = np.nonzero((d >= far[p] + 0.25 * (1 - far[p])) & (n > n[p]))[0]
        q = catch[0] if len(catch) else samples - 1
        bridge = np.linspace(far[p], d[q], q - p + 1)
        d = np.maximum(d, np.concatenate([far[:p], bridge, np.zeros(samples - q - 1)]))
    return n, np.maximum.accumulate(d)


def ghost_grade(depth, rgb, lo, hi, args) -> np.ndarray:
    """Raw depth -> graded uint8 grayscale."""
    n = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    shape_src = n  # undetailed depth, used for relief shading (detail would add stripes)

    # Detail boost: split depth into an edge-preserving base + fine detail, then
    # amplify the detail. Folds, straps and layers separate from each other
    # without halos at silhouettes (the base keeps the big depth jumps).
    if args.detail > 0:
        base = guided_filter(n, n, 30, 1e-3)
        n = np.clip(base + (1 + args.detail) * (n - base), 0, 1).astype(np.float32)

    d = np.interp(n, *tone_curve(args)).astype(np.float32)

    # Self-guided filter: the depth map guides its own smoothing, so surfaces go
    # soft and plastic while real depth jumps (silhouettes) stay razor sharp.
    # Guiding by the photo or adding a blur here makes edges mushy — don't.
    if args.smooth > 0:
        d = d.astype(np.float32)
        d = guided_filter(d, d, args.smooth, 1e-4)

    # Relief: light the depth map like a 3D surface (light from top-left) and
    # multiply it in, so folds and planes catch light and shadow.
    if args.relief > 0:
        surf = cv2.GaussianBlur(shape_src, (0, 0), 1.5) * (400 * shape_src.shape[0] / 2160)
        gy, gx = np.gradient(surf)
        # Cap slopes so silhouette depth jumps don't draw heavy outlines.
        gx, gy = np.clip(gx, -3, 3), np.clip(gy, -3, 3)
        lx, ly, lz = -0.4, -0.5, 0.77
        shade = (-gx * lx - gy * ly + lz) / np.sqrt(gx * gx + gy * gy + 1) / lz
        d = d * (1 - args.relief + args.relief * np.clip(shade, 0, 1.15))

    out =np.clip(d * 255, 0, 255).astype(np.uint8)

    if args.grain > 0:
        noise = np.random.normal(0, args.grain, out.shape)
        out = np.clip(out + noise, 0, 255).astype(np.uint8)
    return out


def guided_filter(I, p, r, eps):
    """Plain-numpy guided filter (He et al.), used if opencv-contrib isn't installed."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = mean(I), mean(p)
    a = (mean(I * p) - mI * mp) / (mean(I * I) - mI * mI + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def pillarbox(gray: np.ndarray, out_w=1920, out_h=1080) -> np.ndarray:
    h, w = gray.shape
    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    canvas[:] = BAR_COLOR
    x, y = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized[..., None]
    return canvas


def finish(gray, args):
    return pillarbox(gray) if args.pillarbox else gray


# ---------------------------------------------------------------- image / video

def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)  # respect phone rotation
    except Exception:
        pass
    return np.array(img.convert("RGB"))


def process_image(path: Path, model: DepthModel, args) -> Path:
    rgb = load_rgb(path)
    if args.height and args.height != rgb.shape[0]:
        # Resize the photo first so depth + edge-aware smoothing happen at final resolution.
        h, w = rgb.shape[:2]
        rgb = np.array(Image.fromarray(rgb).resize(
            (round(w * args.height / h), args.height), Image.LANCZOS))
    depth = model(rgb)
    if args.tta:
        # Average with a mirrored pass: cancels the model's blocky artifacts.
        depth = 0.5 * (depth + model(rgb[:, ::-1].copy())[:, ::-1])
    lo, hi = percentile_range(depth, args.lo_pct, args.hi_pct)
    out = finish(ghost_grade(depth, rgb, lo, hi, args), args)
    dst = Path(args.out) if args.out and not args.batch else path.with_name(f"{path.stem}_depth.png")
    cv2.imwrite(str(dst), out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    if args.save_raw:
        raw = np.clip((depth - depth.min()) / (np.ptp(depth) + 1e-6) * 65535, 0, 65535)
        cv2.imwrite(str(dst.with_name(f"{path.stem}_depth16.png")), raw.astype(np.uint16))
    return dst


def process_video(path: Path, model: DepthModel, args) -> Path:
    dst = Path(args.out) if args.out else path.with_name(f"{path.stem}_depth.mp4")
    cap = cv2.VideoCapture(str(path))  # applies iPhone rotation metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(args.start * fps))
    last = total if not args.seconds else min(total, first + int(round(args.seconds * fps)))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    n_frames = max(last - first, 0)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg is required for video (brew install ffmpeg)")

    cache = Path(args.depth_cache).expanduser() if args.depth_cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    enc = None
    lo = hi = prev = None
    t0 = time.time()
    for i in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        if args.height and args.height != bgr.shape[0]:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (round(w * args.height / h) // 2 * 2, args.height),
                             interpolation=cv2.INTER_AREA if args.height < h else cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Depth cache: the model is the slow part, so keep its raw output per frame.
        # Re-running with different look settings then skips the model entirely.
        cached = cache / f"{first + i:06d}.npy" if cache else None
        if cached and cached.exists():
            depth = np.load(cached).astype(np.float32)
        else:
            depth = model(rgb)
            if cached:
                np.save(cached, depth.astype(np.float16))

        # Hard cut: the new shot shares nothing with the old one, so restart the
        # smoothing instead of easing across the cut (which would ghost the old shot in).
        if prev is not None:
            plo, phi = percentile_range(depth, args.lo_pct, args.hi_pct)
            a_n = np.clip((prev - lo) / max(hi - lo, 1e-6), 0, 1)
            b_n = np.clip((depth - plo) / max(phi - plo, 1e-6), 0, 1)
            if np.abs(a_n - b_n).mean() > args.cut:
                print(f"\ncut detected at frame {first + i}; resetting smoothing", file=sys.stderr)
                lo = hi = prev = None

        # Flicker control, part 1: the black/white range eases toward each new
        # frame instead of jumping (per-frame normalisation pumps brightness).
        flo, fhi = percentile_range(depth, args.lo_pct, args.hi_pct)
        if lo is None:
            lo, hi = flo, fhi
        else:
            a = args.temporal
            lo, hi = a * lo + (1 - a) * flo, a * hi + (1 - a) * fhi
            # Part 2: blend with the previous frame only where depth barely changed
            # (kills shimmer on still surfaces) — moving people get no blend, so no trails.
            if a > 0:
                diff = np.abs(depth - prev) / max(hi - lo, 1e-6)
                w = 0.6 * a * np.exp(-((diff / 0.04) ** 2))
                depth = w * prev + (1 - w) * depth
        prev = depth

        frame = finish(ghost_grade(depth, rgb, lo, hi, args), args)
        frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if enc is None:
            fh, fw = frame.shape[:2]
            fw2, fh2 = fw // 2 * 2, fh // 2 * 2  # H.264 needs even dimensions
            cmd = [ffmpeg, "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw2}x{fh2}", "-r", str(fps), "-i", "-",
                   # Trim the audio input to exactly the rendered frames. (An output -t or
                   # -shortest cut by the audio's length and could drop the last frame.)
                   "-ss", str(args.start), "-t", f"{n_frames / fps:.6f}", "-i", str(path),
                   "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
                   "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac",
                   "-movflags", "+faststart", str(dst)]
            enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        enc.stdin.write(np.ascontiguousarray(frame[:fh2, :fw2]).tobytes())

        done = i + 1
        eta = (time.time() - t0) / done * (n_frames - done)
        print(f"\rframe {done}/{n_frames}  ~{int(eta // 60)}m{int(eta % 60):02d}s left ",
              end="", file=sys.stderr)
    print(file=sys.stderr)
    cap.release()
    if enc is None:
        sys.exit(f"could not read any frames from {path}")
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed while encoding the video")
    return dst


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description="Ghost depth-map effect from a regular photo/video.")
    p.add_argument("input", help="image, video, or folder of images")
    p.add_argument("-o", "--out", help="output path (single file only)")
    p.add_argument("--model", choices=MODELS, help="depth model size (default: large for images, small for video)")
    p.add_argument("--res", type=int, help="model input resolution (default: 770 images, 518 video)")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="where to run the model (auto)")
    p.add_argument("--gamma", type=float, default=1.8, help="falloff into black; higher = darker background (1.8)")
    p.add_argument("--black", type=float, default=0.08, help="black point 0-1; depth below this is pure black (0.08)")
    p.add_argument("--exposure", type=float, default=1.05, help="brightness of the near subject (1.05)")
    p.add_argument("--lo-pct", type=float, default=1, help="percentile mapped to black (1)")
    p.add_argument("--hi-pct", type=float, default=99.5, help="percentile mapped to white (99.5)")
    p.add_argument("--smooth", type=int, default=8, help="edge-aware smoothing radius, 0 = off (8)")
    p.add_argument("--no-tta", dest="tta", action="store_false", help="images: skip the mirrored second pass (2x faster, slightly blockier)")
    p.add_argument("--detail", type=float, default=0, help="amplify folds/layers in the depth, e.g. 1-2 (0 = off)")
    p.add_argument("--relief", type=float, default=0, help="3D surface shading so folds catch light, e.g. 0.3-0.4 (0 = off)")
    p.add_argument("--lift", type=float, default=0, help="bring faint distant shapes out of the black, e.g. 0.35-0.5 (0 = off)")
    p.add_argument("--lift-band", type=float, default=0.35, help="how much of the far range the lift touches, 0-1 (0.35)")
    p.add_argument("--grain", type=float, default=0, help="film grain amount, e.g. 3 (0)")
    p.add_argument("--height", type=int, help="output height in px (images); upscales small photos, e.g. 1080")
    p.add_argument("--pillarbox", action="store_true", help="place on a 1920x1080 frame with dark side bars")
    p.add_argument("--temporal", type=float, default=0.85, help="video flicker damping 0-1 (0.85)")
    p.add_argument("--start", type=float, default=0, help="video: start at this second (0)")
    p.add_argument("--seconds", type=float, help="video: only render this many seconds (quick previews)")
    p.add_argument("--depth-cache", help="video: folder to save/reuse raw per-frame depth (fast re-grades)")
    p.add_argument("--cut", type=float, default=0.12, help="video: scene-cut sensitivity, lower = more cuts (0.12)")
    p.add_argument("--save-raw", action="store_true", help="also save 16-bit raw depth (for Blender/AE displacement)")
    args = p.parse_args()

    src = Path(args.input).expanduser()
    is_video = src.suffix.lower() in VIDEO_EXTS
    model = DepthModel(args.model or ("small" if is_video else "large"),
                       args.res or (518 if is_video else 770), args.device)

    if src.is_dir():
        args.batch = True
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for f in files:
            print(process_image(f, model, args))
    elif src.suffix.lower() in VIDEO_EXTS:
        args.batch = False
        print(process_video(src, model, args))
    else:
        args.batch = False
        print(process_image(src, model, args))


if __name__ == "__main__":
    main()
