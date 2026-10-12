# 5. Carousel stills

`examples/rendezvous/carousel/carousel.py` turns frames of the edit into hero stills in the exact depth-ghost look, plus a final slide.

- One still per character: pick frames where the face isn't hidden (by a mic or a pop filter) and there's headroom.
- It's the same ghost as the video (grade, key light, bloom and mist, grain, vignette). On a close-up still, the focus sits on the face, since a still has no time for a focus pull.
- Leave anamorphic flare streaks off stills: on a portrait they cut across the face.
- Sizes: **1080×1920** (TikTok photo mode) and **1080×1350** (Instagram 4:5).
- Final slide: a blackletter headline, the repo link in a pill, and "made with Claude Code". Keep text in the middle of a 9:16 slide, clear of the caption area.

```bash
python carousel.py out/      # -> out/9x16/*.png, out/4x5/*.png
```

To try it on your own photo instead, run `/depth-ghost` on it ([guide 1](01-picture.md)) and crop to 4:5 or 9:16.
