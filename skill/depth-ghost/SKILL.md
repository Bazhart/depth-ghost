---
name: depth-ghost
description: Turn a regular photo (iPhone HEIC, JPG, PNG), video, or folder of images into the "ghost depth" 3D look — near things glow white, far things fall into black, like a depth scan (monocular depth map via Depth Anything V2). Use when the user types /depth-ghost, or asks for a depth map, ghost depth effect, depth 3D effect, or "make this look like a depth scan" on an image or video.
---

# Depth Ghost

Everything lives in this skill's folder (the "Base directory for this skill" shown above — call it `SKILL_DIR`):

- `SKILL_DIR/scripts/depth_ghost.py` — the effect (photos, folders, quick video previews)
- `SKILL_DIR/scripts/depth_ghost_smooth.py` — **finished videos**: same look, plus shot-aware temporal smoothing (see "Smooth video mode" below)
- `SKILL_DIR/scripts/video_metrics.py` — measures how steady a rendered video is (pumping, shimmer, jitter), per shot
- `SKILL_DIR/scripts/setup.sh` — one-time install of a private Python env at `~/.depth-ghost/.venv`
- Python to run it with: `~/.depth-ghost/.venv/bin/python` (never the system Python — it usually has no torch, or a version too new for torch)

## The look (what "good" means here)

Pure glowing depth: the subject glows white, surfaces are soft and smooth, silhouettes are sharp, the background melts into black. No outlines, no shading, no halos. Folds and layers of clothing should read as different shades of grey, not as one flat white blob.

## Settings

The user may give any of these as numbers or in plain words. Anything they don't mention uses the default.

| Setting | Flag | Default | What it does |
|---|---|---|---|
| model | `--model small\|base\|large` | large (images), small (video) | Bigger = cleaner edges, much slower. |
| resolution | `--res N` | 770 (images), 518 (video) | Model input size (multiple of 14). 518 fast, 1036 finest edges. |
| device | `--device auto\|cpu\|mps\|cuda` | auto | auto picks GPU on Apple Silicon/NVIDIA, CPU on Intel Macs (faster there). |
| output height | `--height N` | source size | Resize the photo/video frames to N px tall BEFORE depth (e.g. 1080, 2160). Use when the source is small or the user names a size; never upscale the result afterwards (looks mushy). |
| gamma | `--gamma X` | 1.8 | Falloff into black. Higher = darker background AND more tonal separation on the subject (1.5–3). |
| black point | `--black X` | 0.08 | 0–1. Far depths below this become pure black. |
| exposure | `--exposure X` | 1.05 | How bright the near subject glows. Highlights roll off softly, so 1.5–2 glows without clipping folds. |
| detail | `--detail X` | 0 | Amplifies folds, straps, layers inside the depth. 1.5 is good; >2.5 starts to look noisy. |
| background lift | `--lift X` | 0 | Brings far shapes (buildings, domes, skylines) out of the black as faint ghosts; sky stays black, subject untouched. 0.35 subtle, 0.5 clear. |
| lift band | `--lift-band X` | 0.35 | How much of the far range the lift touches. |
| smoothing | `--smooth N` | 8 | Self-guided filter: soft surfaces, sharp silhouettes. 0 = off, 14+ = very soft. |
| mirror pass | `--no-tta` | on | Images average a mirrored second pass (cleaner, 2× time). `--no-tta` to skip. |
| relief | `--relief X` | 0 | 3D surface shading. Draws a dark outline around subjects — only use if the user explicitly asks for shading/relief. |
| grain | `--grain X` | 0 | Film grain, e.g. 3. |
| pillarbox | `--pillarbox` | off | Put on a 1920×1080 frame with dark side bars. |
| percentiles | `--lo-pct`, `--hi-pct` | 1, 99.5 | Depth range mapped to black/white. |
| flicker damping | `--temporal X` | 0.85 | Video only. Higher = steadier; moving subjects never trail (still areas only). 0 = off. |
| start | `--start S` | 0 | Video only: begin at second S. |
| depth cache | `--depth-cache DIR` | off | Video only: saves raw per-frame depth; re-runs with different look flags skip the model (minutes instead of hours). Always use it for full renders. |
| cut sensitivity | `--cut X` | 0.12 | Video only: hard cuts reset the flicker smoothing (prints "cut detected"). Lower = more cuts. |
| seconds | `--seconds S` | whole clip | Video only: render just S seconds (previews). |
| raw 16-bit | `--save-raw` | off | Also save 16-bit depth (for Blender / After Effects displacement). |
| output | `-o path` | `<name>_depth.png/.mp4` next to input | Single file only. |

### Recipes (plain words → flags)

- **Default / classic ghost** → no flags.
- **"show the folds / layers", "looks flat"** → `--gamma 2.5 --exposure 1.7 --detail 1.5`
- **Far background disappears (city, buildings, monument)** → add `--lift 0.45`. "More/less background" → nudge by ±0.05–0.1.
- **"darker background"** → gamma +0.5. **"subject brighter"** → exposure +0.3. **"softer"** → smooth 12. **"sharper"** → smooth 4.
- **Small source (< ~1000 px tall) or a target size given** → `--height 1080` (or the size asked).
- **iPhone photo** → `--height 2160 --res 1036` for full quality.
- **Video** → same look flags as photos; model/res default to small/518 automatically. Audio is kept, iPhone .MOV rotation is handled, output is H.264 .mp4.

### Video workflow

Video is slow: every frame runs the depth model. Rough speed per frame with the default small model: ~0.2–0.5 s on Apple Silicon, ~1.5–2.5 s on Intel Macs (×30 frames per second of footage). `base` is ~3× slower, `large` ~10×+ — only suggest them for short clips.

1. Tell the user the estimated time (frames = seconds × fps; check with `ffprobe`).
2. **Always render a 2–3 s preview first** (`--seconds 3`, pick a representative `--start`), pull 3 frames with ffmpeg (`-vf "select='eq(n\,0)+eq(n\,35)+eq(n\,70)',scale=480:-1,tile=3x1"`), look at them, show the user.
3. Once the look is approved, run the full clip **in the background** with `depth_ghost_smooth.py` (see "Smooth video mode"; it prints one line per shot), and report when done.
4. Lift is for far backgrounds (buildings, skylines). On scenes that are mostly mid-distance (crowds, rooms) it greys everything out — leave it off there.

## Smooth video mode (use for every finished video)

`depth_ghost.py` grades each frame on its own, so the result flickers: the model re-guesses its depth scale every frame (the whole image pumps), regions pulse, edges jitter. `depth_ghost_smooth.py` fixes that and treats **every hard cut as a new clip** — nothing is smoothed, blended or normalised across a cut, each shot gets its own range and its own smoothing.

Per shot, in order: steady the overall depth scale (anchored to a smoothed curve, no drift) → motion-compensated consensus filter (±3 frames warped by optical flow from the footage; one-frame model hiccups are outvoted; only the broad correction is kept so silhouettes stay crisp) → still-area blend (zero-phase, only where depth barely changes — movers are never smeared) → one fixed black/white range for the shot → grade → regional deflicker (6×4 grid, measured only on pixels that are still in the footage, ±6% max).

```bash
~/.depth-ghost/.venv/bin/python "SKILL_DIR/scripts/depth_ghost_smooth.py" "<video>" -o "<out>.mp4" \
  --model base --res 770 --smooth 0 <look flags> 2>&1 | grep -E "shot|frames ->|rror"
```

- Takes all `depth_ghost.py` look flags, plus:
  - **Highlight shoulder** — `--knee 0.70 --tail 0.06` keeps the nearest object (mic, railing, door frame) from blowing out to white while keeping its internal gradation; the subject below the knee is untouched. To dim it further lower `--bend` (e.g. 0.04); lowering `--knee` also dims it but starts flattening the subject. (Without `--tail`: `--knee K --peak P` = classic roll-off toward P.)
  - **Shots** — `--cuts auto` (default; a cut = one-frame spike in pixel change *and* a colour-histogram change, so whip pans and motion blur are not cuts), `--cuts none`, or `--cuts 305,319,538` (frames where new shots start). Check the printed shot list against a contact sheet; very short shots (a few frames) are real and fine.
  - Smoothing knobs (defaults are the tested ones): `--stab-k 3`, `--still-cap 0.8`, `--deflicker-grid 6x4`, `--deflicker-max 0.06`; each `0` turns that stage off.
- Depth is cached per frame (`~/.depth-ghost/cache/<name>_<model>_<res>` unless `--depth-cache`), and the smoothed depth next to it (keyed by the smoothing settings and shots). **Re-grading with new look flags skips the model and the smoothing** — minutes, not the full render time. Reuse an existing cache from a `depth_ghost.py --depth-cache` run by passing the same dir.
- Speed (M4, base/770): model ~1.3 s/frame (first run only), smoothing ~0.6 s/frame (first run per setting), grade ~0.2 s/frame.
- Memory: a whole shot is held in RAM (~10 MB per 1080p frame); fine for shots up to ~25 s on 16 GB.
- **Verify:** `~/.depth-ghost/.venv/bin/python "SKILL_DIR/scripts/video_metrics.py" "<source>" "<render>" [other renders] --per-shot` — lower is steadier. Typical: plain grade pump ~0.46/255, ~20 visible jumps per 8 s; smooth mode ~0.29, ~5 jumps. Always also look at a zoom of fast motion (arms, hands) for specks or blotches.

### Smoothing lessons (tested, don't repeat)

- Wider motion window (±6) or a motion-adaptive window: smoother numbers, but blotches/specks on fast **textureless** motion (a white sleeve) — flow misplaces it and the colours still match. Keep ±3.
- Stronger still-blend (cap > 0.85): distant slow walkers fade away. Gating the blend on footage stillness: mottled patches on clothing. Keep cap 0.8.
- Whole-frame (1×1) deflicker: when the subject moves the frame average changes legitimately, and correcting it makes the static background shimmer. Use the regional grid, measured on still pixels.
- Frame-to-frame scale alignment chained over time drifts and creates new jumps; anchor to a smoothed curve instead.
- Higher model res (1036) did not add visible detail on 1080p footage and was less stable; 770 is the sweet spot. Detail tops out around `--detail 2.5`.
- Background: `--lift 0.30` brings walls/stairs/gear out as faint ghosts in rooms while people stay separated; 0.45 washes rooms grey.

## Steps

1. **Setup (first use only):** run `bash "SKILL_DIR/scripts/setup.sh"`. It exits instantly if already installed. The first real run also downloads the model (~1.3 GB for large).
2. **Input:** attached images live in the session's images folder — copy them next to where outputs should go (e.g. a `depth-ghost` folder in the working directory) with a short descriptive name. HEIC works directly. For an aspect ratio request (e.g. 1:1), crop the source with PIL first, centered on the subject, leaving out near objects at frame edges (door frames, walls, panels) so the subject gets the white.
3. **Run** with a long timeout (large model ≈ 1–3 min per image; big portrait photos may print "GPU out of memory; falling back to CPU" — expected, it still works):
   ```bash
   ~/.depth-ghost/.venv/bin/python "SKILL_DIR/scripts/depth_ghost.py" "<input>" <flags> 2>&1 | tail -1
   ```
   For photos, when iterating on look-only settings, compute depth once (import the script, call `DepthModel` then `ghost_grade`) and regrade, instead of rerunning the model each time.
4. **Check it yourself:** build a side-by-side (original | result, same height) and look at it. Is the subject the brightest thing? Any outlines or halos? Folds readable or clipped flat? Background visible where it should be?
5. **Show the user** the result and the comparison, and report the flags used. If something near the camera that isn't the subject (door frame, wall edge) steals the white, say so and suggest one concrete fix (crop it out, or raise exposure). One tweak, not a list.

### Video lessons (from real jobs)

- **From a YouTube link:** `yt-dlp --download-sections "*START-END" --force-keyframes-at-cuts -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" --merge-output-format mp4 -o source.mp4 URL` — downloads only the needed seconds.
- **Look before choosing settings:** make a contact sheet (`ffmpeg -vf "fps=0.8,scale=480:-1,tile=4x2"`) and detect cuts (`select='gt(scene,0.3)',showinfo`). Note what is nearest to camera — foreground railings/frames take the white and push people to dim grey.
- **People behind a near foreground object:** use a flatter curve so mid-distance people stay readable: `--gamma 1.4 --exposure 1.3 --detail 1.5`, no lift.
- **"Sharp, no blur":** `small` gives soft mushy silhouettes. Use `--model base --res 770 --smooth 0` (sharpest clean edges; large isn't sharper). Do NOT try edge-snapping (dilate/erode toggling) — it draws dark outlines.
- **Pick the look on 3 sample frames** (compute depth for them once, regrade with `ghost_grade` in different settings), then run the full clip with `--depth-cache`; regrade from the cache if the look needs a tweak.
- **Verify the final video:** frame strip across the clip, check frame count/fps/audio with ffprobe, and measure flicker (mean frame-to-frame brightness change; under ~1.5/255 is steady). Check the frames right after any cut for ghosting.

