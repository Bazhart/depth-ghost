# 2. Make a video

```
/depth-ghost ~/Desktop/clip.mov
```

Claude estimates the time, renders a **3-second preview** first, shows you 3 frames, and only runs the full clip once you like the look. Audio is kept; iPhone rotation is handled; the output is H.264 MP4.

## Why "smooth mode"

A depth model guesses every frame on its own, so a plain render pumps and flickers: the whole image breathes, regions pulse, edges jitter. `depth_ghost_smooth.py` fixes this, **shot by shot**. Every hard cut is treated as a new clip, and nothing is smoothed across a cut. Inside each shot it:

1. steadies the overall depth scale (anchored to a smoothed curve, no drift);
2. runs a motion-compensated consensus filter (±3 frames warped by optical flow, so one-frame model hiccups get outvoted);
3. blends still areas over time (movers are never smeared);
4. locks one black/white range for the shot;
5. grades, then deflickers by region, measured only on pixels that are still.

## The command

```bash
~/.depth-ghost/.venv/bin/python ~/.claude/skills/depth-ghost/scripts/depth_ghost_smooth.py clip.mov \
  -o clip_ghost.mp4 --model base --res 770 --smooth 0 <look flags>
```

The look used for **RENDËZVOUS** (a 47 s edit of 8 shots in a dark studio):

```bash
--model base --res 770 --gamma 1.4 --black 0.07 --exposure 1.4 --detail 2.5 --smooth 0 \
--temporal 0.95 --hi-pct 99.2 --knee 0.70 --tail 0.06 --bend 0.07 --lift 0.15 \
--depth-cache ~/.depth-ghost/cache/my_clip
```

- `--knee 0.70 --tail 0.06 --bend 0.07`: a highlight shoulder, so the nearest object (a mic, a railing) keeps its gradation instead of blowing out white.
- `--lift 0.15`: the room comes out of the black just a little, while people stay separated.
- `--cuts auto` is the default (a cut = a one-frame jump in pixels **and** colours, so whip pans aren't cuts). You can also pass `--cuts 305,319,538`.
- `--depth-cache DIR`: re-grading with new look flags then skips the model and the smoothing, so a tweak takes minutes instead of hours.

## Speed

Base model at 770 on an M-series Mac: ~1.3 s/frame for the model (first run only), ~0.6 s/frame for smoothing, ~0.2 s/frame for the grade. A 47 s clip at 30 fps is about 1,400 frames, so plan a long first render and fast re-grades.

## Check it

```bash
~/.depth-ghost/.venv/bin/python ~/.claude/skills/depth-ghost/scripts/video_metrics.py clip.mov clip_ghost.mp4 --per-shot
```

Lower is steadier. Also look at a zoom of fast motion (hands, sleeves) for specks.

## From YouTube

```bash
yt-dlp --download-sections "*0:10-0:20" --force-keyframes-at-cuts \
  -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" --merge-output-format mp4 -o source.mp4 URL
```

Only use footage you have the rights to.
