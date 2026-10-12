#!/usr/bin/env python3
"""QA for the TikTok clips: specs, audio vs the song, loop seams (picture and sound), contact sheets."""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import tiktok_cut as tc
files = [Path(f) for f in sys.argv[1:]]
for f in files:
    key = f.name.split("_")[2]
    clip = tc.Clip(key, tc.CLIPS[key])
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(f)], capture_output=True, text=True).stdout)
    v = next(s for s in pr["streams"] if s["codec_type"] == "video"); a = next(s for s in pr["streams"] if s["codec_type"] == "audio")
    nfr = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.strip())
    spec = f"{v['width']}x{v['height']} {v['r_frame_rate']} {v['codec_name']} {v.get('profile')} {v.get('pix_fmt')} {v.get('color_space')} | {nfr} frames (want {clip.N}) | {float(pr['format']['duration']):.3f}s | {int(pr['format']['bit_rate']) / 1e6:.1f} Mbps | {a['codec_name']} {a['sample_rate']} Hz {a['channels']} ch"
    au = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-vn", "-ac", "1", "-ar", "44100", "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
    ref = tc.clip_audio(clip).mean(1)
    refst = tc.clip_audio(clip)
    n = min(len(au), len(ref)) - 4410
    corr = np.corrcoef(au[2205:n], ref[2205:n])[0, 1]
    lv = lambda x: 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
    st = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-vn", "-ac", "2", "-ar", "44100", "-f", "f32le", "-"], capture_output=True).stdout, np.float32).reshape(-1, 2)
    m = min(len(st), len(refst)) - 4410
    line = f"{f.name}\n   {spec}\n   audio vs song segment: corr {corr:.4f}, stereo level {lv(st[2205:m]):.2f} vs {lv(refst[2205:m]):.2f} dBFS, length {len(au) / 44100:.3f}s"
    cap = cv2.VideoCapture(str(f)); frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(cv2.resize(fr, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32))
    d = np.array([np.abs(frames[i] - frames[i - 1]).mean() for i in range(1, len(frames))])
    if clip.loop:
        seam = np.abs(frames[0] - frames[-1]).mean()
        cuts = set(clip.starts.tolist())
        within = [d[i - 1] for i in range(1, len(frames)) if i not in cuts and d[i - 1] > 0]
        aseam = abs(au[0] - au[len(ref) - 1]) if len(au) >= len(ref) else float("nan")
        adiff = np.abs(np.diff(au[:len(ref)]))
        line += (f"\n   LOOP seam picture: last->first diff {seam:.2f} (inside shots: median {np.median(within):.2f}, 95th pct {np.percentile(within, 95):.2f}, cuts median {np.median([d[c - 1] for c in cuts if c > 0]):.2f})"
                 f"\n   LOOP seam sound: jump {aseam:.4f} (sample-to-sample: median {np.median(adiff):.4f}, 99.9th pct {np.percentile(adiff, 99.9):.4f}); decoded audio {len(au)} vs exact {len(ref)} samples")
    print(line, flush=True)
    step = max(1, len(frames) // 28)
    sel = frames[::step][:28]
    while len(sel) % 7: sel.append(np.zeros_like(frames[0]))
    tiles = [cv2.putText(s.astype(np.uint8).copy(), f"{k * step}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) for k, s in enumerate(sel)]
    sheet = np.vstack([np.hstack(tiles[r:r + 7]) for r in range(0, len(tiles), 7)])
    cv2.imwrite(str(HERE / "qa" / f"clip_{key}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
