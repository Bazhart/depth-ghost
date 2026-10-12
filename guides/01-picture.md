# 1. Make a picture

## The easy way (in Claude Code)

```
/depth-ghost
```

Attach a photo (JPG, PNG, iPhone HEIC). Claude runs the depth model, builds a side-by-side with your original, checks it and shows you the result. Then just say what you want, in plain words:

```
/depth-ghost show the folds, bring the background back
/depth-ghost darker background, subject brighter
/depth-ghost 1:1 crop, 1080
/depth-ghost gamma 2.5, exposure 1.7, detail 1.5, lift 0.45
```

## What the settings do

| Say | Flag | Default | Effect |
|---|---|---|---|
| "darker background" | `--gamma` | 1.8 | Falloff into black. Higher = darker background and more separation on the subject (1.5–3). |
| "brighter subject" | `--exposure` | 1.05 | How hard the near subject glows; highlights roll off softly (1.5–2 glows without clipping). |
| "show the folds" | `--detail` | 0 | Folds, straps, layers inside the depth. 1.5 is good, past 2.5 gets noisy. |
| "bring the background back" | `--lift` | 0 | Far shapes (buildings, walls) come out of the black as faint ghosts. 0.3 subtle, 0.5 clear. |
| "softer" / "sharper" | `--smooth` | 8 | Soft plastic surfaces with sharp silhouettes. 4 sharper, 12 softer, 0 off. |
| "crush the black" | `--black` | 0.08 | Depths below this become pure black. |
| size | `--height` | source | Render small photos bigger *before* depth (never upscale afterwards). |
| quality | `--model`, `--res` | large, 770 | iPhone photos: `--height 2160 --res 1036`. |
| grain | `--grain` | 0 | Film grain, e.g. 3. |

**Quick recipes**

- Classic ghost: no flags.
- "Looks flat": `--gamma 2.5 --exposure 1.7 --detail 1.5`.
- City or monument behind: add `--lift 0.45`.
- People behind something near the camera (a railing, a mic): `--gamma 1.4 --exposure 1.3 --detail 1.5`.

## Command line

```bash
~/.depth-ghost/.venv/bin/python ~/.claude/skills/depth-ghost/scripts/depth_ghost.py photo.jpg \
  --gamma 2.5 --exposure 1.7 --detail 1.5            # -> photo_depth.png
```

A folder works too (every image inside). `--save-raw` also writes 16-bit depth for Blender / After Effects displacement.

## Tips

- The nearest thing gets the white. If a door frame or a mic steals it, crop it out first (or raise exposure).
- For a square or vertical post, crop the *source* around the subject before running.
- Want stills from a video you already rendered? See [5. Carousel stills](05-carousel.md).
