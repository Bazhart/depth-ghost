# 4. TikTok clips & perfect loops

`examples/rendezvous/tiktok/tiktok_cut.py` renders vertical clips (1080×1920, 30 fps) from the same plates, each on its own moment of the song. The edit lists are in `tiktok_edl.py`.

## Pick the moments

- `song_map.py` finds the bar grid and the **8-bar block that repeats most**, which is the hook. In RENDËZVOUS it repeats 3 times, each right on an 808 drop.
- `song_events.py` lists the hits, triplets and breaks for the whole song, so each clip gets cuts that land on its own music.
- Good TikTok windows: break → drop, a full hook, a verse with a dropout, the outro to the end of the song.

## Framing

- The vertical camera tracks the singer and **anchors every zoom on the face**, so the face stays in the upper part of the frame, clear of TikTok's caption area at the bottom and the buttons on the right.
- Keep the ghost *exactly* as in the master. Re-use the same grade function; don't "improve" it for vertical.

## Clean person masks

Depth bands make bad masks in busy rooms: half bodies, floor patches, walls. What worked:

1. **Apple Vision person segmentation**, built into macOS, run from JavaScript for Automation (no compiler needed), on **portrait crops around the singer**. On the full wide frame it often returns nothing.
   ```bash
   osascript -l JavaScript tools/personseg.js CROPS_DIR MATTES_DIR 0:305,319:538
   ```
   (one range per shot: the request is stateful, so mattes stay steady inside a shot and restart at each cut)
2. `clean_mattes.py`: map the mattes back into the frame, take a 5-frame temporal median, keep only the singer's own shape, drop anything far behind him, and pull the edge slightly inside the silhouette (the ghost's own outline becomes a rim light instead of a colour fringe).
3. Measure the frame-to-frame IoU, and **only use cut-outs where it stays above 0.9**. Check every masked segment by eye; mics and pop filters are where it fails.
4. Fade layered elements out well before their own edges, so there's never a floating cut-out with a hard bottom line.

## Perfect loops

- Loop exactly **8 bars**. At 140 BPM that's 13.714 s = 411.43 frames, so fit the audio to **411 whole frames** with a periodic (FFT) resample: 0.1 % faster, inaudible.
- Crossfade the last 10 ms into what *really* precedes the first downbeat in the song, so the wrap sounds like the song itself.
- Make the last shot run on into the first one (same footage, same camera move, light trails carried over), and solve the camera on a doubled timeline.
- Encode the loop's audio with ffmpeg's **native `aac`**, which writes an exact edit list so the audio length equals the picture to the sample. `aac_at` pads about 23 ms, which leaves a hiccup on every repeat.
- Best seams: a hook whose last triplet crashes on the restart, or a hook that ends in a break and slams back into its own first beat.

## QA before posting

```bash
python qa_clips.py render/*.mp4      # specs, audio vs song, loop seams, contact sheets
python ../flash_check.py render/*.mp4
```

Specs that work: H.264 High, yuv420p, ~20 Mbps (so files stay under phone upload limits), AAC 320k.

When posting, keep the original sound, and don't trim the loops in TikTok's editor.
