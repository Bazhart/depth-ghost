# 3. Cut a music video around it

The ghost render is the base layer. What makes it a music video is everything keyed to the song on top of it. All of this was built by talking to Claude Code; the resulting scripts are in [`examples/rendezvous`](../examples/rendezvous) (`final_cut.py` is the main edit, `remix_cut.py` the re-edit).

## The plates

1. Render the ghost with smooth mode ([guide 2](02-video.md)) and **keep the depth cache**.
2. Extract the colour footage and the ghost render as frames (`frames/colour`, `frames/ghost`), so any frame can be read at random, which re-edits and slow-mo need.
3. Replace the audio with your track: lined up at 0, cut at the end, full level. On a Mac `-c:a aac_at -b:a 320k` keeps the level identical to your WAV.

## The look

- **Duotone grade:** near-black → steel blue → icy white through a LUT with a soft S-curve and a highlight shoulder. Keep real blacks, or the ghost loses its separation.
- **Key light:** a soft light that follows the singer and lifts only things at his depth, never the wall behind.
- **Light:** long-exposure trails (highlights leave a decaying, spreading trail), mist and bloom, and anamorphic flares built only from the **real** lights found in the colour footage. Skip lens-ghost dots; they look cheap.

## Keyed to the song

- **Beat map:** detect the 808 + open-hat hits (onsets in the sub band and the 6–14 kHz band within 35 ms of each other), the "dun dun dun" triplets (three evenly spaced hits), and the **breaks** where the 808 drops out. `examples/rendezvous/tiktok/song_events.py` does this for a whole song.
- **Handheld camera:** seeded noise per shot, a bob on the beat, a punch-zoom on every hit, *step, step, crash* on the triplets, and a zoom that lands **exactly on the frame** the 808 comes back after a break. Check the drop by measuring the sub level, not by ear; mine was 0.5 s off the first time.
- **Colour punches:** switch from ghost to the real footage for 3–10 frames on impacts, then hand back to the ghost nearest-first.
- **Depth rings:** a contour races from near to far after each hit. A **glitch** fires when the 808 drops out. Add an exposure kick and a touch of chromatic aberration on impacts.

## Time

- **Posterize time:** the finished frame updates only about 12 times a second, re-anchored on every hit so hits still land on their exact frame.
- **Speed ramps:** 0.2× slow-mo on crashes, 3× rushes through breaks, with frame blending in slow-mo. Keep each shot on its own footage so cuts stay on the beat.

## Transitions (one different per cut)

Depth sweep (the new shot appears nearest-first behind a scan line), far-first, whip with motion blur, depth dissolve, defocus, scan line, punch-in.

## Re-edit / layering (`remix_cut.py`)

- An **edit list** (start frame, shot, source frame, speed, options) lets you break up long holds with cutaways on the beat.
- **Loom:** another shot's subject, depth-masked, double-exposed into the room.
- **Matte:** the real singer in colour, cut out by depth, inside the ghost room.
- **Triptych:** three panels landing on the half-beats.

## Before you post: flash safety

```bash
python examples/rendezvous/flash_check.py my_video.mp4
```

This is a WCAG 2.3.1 check: no more than 3 flashes in any 1 second, on the whole frame and on a 4×4 grid of regions. Strobes and colour punches add up fast, so thin them wherever a window hits 3.
