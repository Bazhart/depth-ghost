# RENDËZVOUS: reference scripts

These are the exact scripts behind the RENDËZVOUS edit. They are tied to that footage: frame numbers in the edit lists, singer positions marked by eye, the song's measured hits. Read them as a worked example and adapt them to your own clip (or ask Claude Code to do it with you). **No footage, audio or renders are included.**

| File | What it does |
|---|---|
| `final_cut.py` | The main edit ("Scan Cut"): grade, key light, beat-keyed handheld camera, transitions, light trails, flares, colour punches, rings, glitches, posterize time, speed ramps. Renders 16:9 (2.39 letterbox), 9:16 and 1:1. |
| `remix_cut.py` | The re-edit: an edit list with random-access frames, cutaways on the beat, loom / matte / triptych layering, a break strobe. |
| `remix_social.py` | The re-edit in 9:16 and 1:1. |
| `flash_check.py` | WCAG 2.3.1 flash check (≤ 3 flashes per second, whole frame + 4×4 regions). |
| `tiktok/tiktok_cut.py` | The 10 vertical TikTok clips: face-anchored camera, output-space transitions, seamless 8-bar loops. Edit lists in `tiktok_edl.py`. |
| `tiktok/song_map.py` | Bar grid + the most-repeated 8-bar block (the hook). |
| `tiktok/song_events.py` | 808 + open-hat hits, triplets, breaks, drops for the whole song. |
| `tiktok/tools/personseg.js` | Apple Vision person mattes via JavaScript for Automation (macOS, no compiler). `personseg.swift` is the same thing in Swift. |
| `tiktok/clean_mattes.py` | Mattes back into the frame, temporal median, singer-only, edge pulled in, stability report. |
| `tiktok/qa_clips.py` | Specs, audio vs song, loop seams (picture + sound), contact sheets. |
| `carousel/carousel.py` | Hero stills + final slide (9:16 and 4:5). |

## What they expect

- The skill's Python environment: `~/.depth-ghost/.venv/bin/python`
- `source.mov` (or `DG_SOURCE=...`): your edit
- `ghost_base_lift15.mp4`: the smooth-mode ghost render of it ([guide 2](../../guides/02-video.md))
- `DG_DEPTH_CACHE`: the smoothed depth cache that render wrote
- `frames/colour/000000.jpg…` and `frames/ghost/000000.jpg…`: both extracted as frames (`ffmpeg -i source.mov -q:v 2 frames/colour/%06d.jpg`)
- `audio/track.wav` (or `DG_AUDIO=...`): your song, starting at 0
- `beats.txt`: the beat grid (first line `bpm`, second `downbeat_phase`, then one time per line)
- `fonts/UnifrakturMaguntia-Book.ttf` (Google Fonts, OFL) for the blackletter title and slide
