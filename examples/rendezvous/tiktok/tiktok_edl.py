"""
Edit lists for the ten TikTok clips. Song bars are FL Studio bars at 140 BPM (bar 0 = 0 s; the hook "i walk on
holy water" starts at bars 8, 32 and 64). Inside a clip, positions are in bars from the clip's start.

Segment: (bar, shot, source frame, speed, options). Each runs until the next one.
options: tr=(kind, frames) transition in | bleed | iso=1 (singer in colour, only inside the verified matte
windows, see ISO_OK) | loom=(shot, src, speed, opacity, dx of frame width, scale) | tri=[(shot, src) x3] |
stutter=k | freeze=frames | calm | hold=k (posterize) | e / z / cy / comp (camera)
clip keys: bars, loop, scan=[(bar, length, "in"|"out")], fast=[(bar, bar)] posterize 10 fps, fade_out frames,
punch=[bars] extra colour punches (drops and triplet crashes get them automatically).

Shots: A wide at the mic, B stairs flash, C whip pans, D stairs set, E dancer, F close-up, G wide at the mic
(second angle), H hoodie enters.
"""

# Frames where the Vision person matte is temporally stable (IoU >= 0.9 frame to frame, steady area);
# F uses its depth band. Colour isolation is only allowed inside these.
ISO_OK = {"A": [(27, 64), (233, 287)], "E": [(888, 919)], "F": [(919, 1151)], "G": [(1182, 1318)]}

CLIPS = {
    # 1. Intro -> drop. Opens on the real footage; the depth scanner turns the room into the ghost; the intro
    #    triplet crashes into a slow-mo arm sweep; the 808 break floats on the close-up; the drop slams.
    "01": dict(name="scan_reveal", bars=(0, 10), scan=[(0.5, 1.5, "in")], edl=[
        (0.00, "A", 20, 1.0, {"calm": True, "e": 0.6}),
        (2.00, "G", 1160, 1.0, {"tr": ("whip", 8)}),
        (3.02, "E", 790, 1.0, {}),
        (3.27, "C", 470, 1.0, {}),
        (3.52, "A", 179, 0.30, {}),
        (4.52, "D", 650, 1.0, {"tr": ("sweep", 12)}),
        (5.52, "F", 930, 1.0, {"tr": ("defocus", 10)}),
        (6.83, "H", 1340, 1.0, {}),
        (7.02, "E", 860, 1.0, {}),
        (7.27, "F", 1005, 0.35, {}),
        (8.00, "E", 870, 0.30, {}),
        (8.53, "G", 1240, 1.0, {}),
        (9.02, "A", 240, 1.0, {}),
    ]),
    # 2. LOOP: hook 1, 8 bars. The closing triplet crashes on the restart inside one continuous shot (G).
    "02": dict(name="LOOP_holy_water", bars=(8, 16), loop=True, edl=[
        (0.00, "G", 0, 1.0, {}),                             # = the last segment running on through the seam
        (1.02, "E", 740, 1.0, {}),
        (2.02, "A", 40, 1.0, {}),
        (2.89, "D", 660, 1.0, {}),
        (3.77, "H", 1345, 1.0, {}),
        (3.90, "C", 450, 1.0, {}),
        (4.02, "F", 1000, 0.30, {"bleed": True}),
        (5.01, "E", 860, 1.0, {"tr": ("punch", 8)}),
        (6.02, "A", 175, 0.5, {}),
        (6.89, "F", 1060, 1.0, {}),
        (7.51, "G", 1200, 1.0, {}),
    ]),
    # 3. Verse 1 cut-up: half-bar cuts, step-printing, double triplet, a strobe through the 808 break, slam.
    "03": dict(name="verse_cutup", bars=(16, 25), fast=[(0, 8)], edl=[
        (0.00, "E", 760, 1.0, {"stutter": 2}),
        (0.53, "A", 30, 1.0, {}),
        (1.00, "C", 400, 1.0, {}),
        (1.50, "G", 1170, 1.0, {"stutter": 2}),
        (2.00, "D", 670, 1.0, {}),
        (2.39, "F", 940, 1.0, {}),
        (2.64, "E", 820, 0.3, {}),
        (2.78, "H", 1330, 1.0, {}),
        (3.02, "A", 200, 0.3, {}),
        (4.02, "G", 1230, 1.0, {"tr": ("scanline", 10)}),
        (4.46, "E", 880, 1.0, {}),
        (5.02, "F", 1010, 1.0, {"bleed": True}),
        (6.02, "C", 500, 1.0, {}),
        (6.21, "D", 700, 0.3, {}),
        (7.02, "A", 250, 1.0, {}),
        (7.62, "G", 1210, 1.0, {}),
        (7.80, "F", 990, 1.0, {}),
        (8.00, "E", 895, 0.25, {}),
        (8.52, "H", 1370, 1.0, {}),
    ]),
    # 4. The long 808 dropout before hook 2: slow-mo close-up, focus pull, colour bleeding in; the drop slams.
    "04": dict(name="dropout", bars=(28, 34), edl=[
        (0.00, "G", 1185, 1.0, {}),
        (1.00, "E", 750, 1.0, {}),
        (2.02, "A", 236, 1.0, {}),
        (2.50, "F", 972, 0.30, {"bleed": True, "calm": True, "hold": 2}),
        (4.00, "E", 865, 0.30, {}),
        (4.52, "C", 470, 1.0, {}),
        (5.02, "G", 1250, 1.0, {"iso": 1}),
        (5.51, "A", 160, 1.0, {}),
    ]),
    # 5. LOOP: hook 2, 8 bars. Ends in its 808 break and slams back into its first downbeat; the dancer shot
    #    runs through the seam.
    "05": dict(name="LOOP_hook_two", bars=(32, 40), loop=True, edl=[
        (0.00, "E", 0, 1.0, {}),                             # = the last segment running on through the seam
        (1.02, "A", 233, 1.0, {"iso": 1}),
        (2.03, "G", 1165, 1.0, {}),
        (2.26, "C", 420, 1.0, {}),
        (2.39, "D", 690, 1.0, {}),
        (2.52, "F", 1020, 0.35, {"bleed": True}),
        (3.77, "H", 1335, 1.0, {}),
        (3.89, "A", 120, 1.0, {}),
        (4.02, "E", 890, 0.30, {"iso": 1}),
        (5.02, "G", 1262, 1.0, {"tr": ("depth_dissolve", 10)}),
        (6.02, "F", 1090, 1.0, {}),
        (7.02, "A", 165, 1.0, {}),
        (7.63, "E", 790, 1.0, {}),
    ]),
    # 6. 808 chaos: four triplets, a rushed break with rolls, glitches, step-printing, slam.
    "06": dict(name="808_chaos", bars=(44, 49), fast=[(0, 4)], edl=[
        (0.00, "E", 730, 1.0, {}),
        (0.26, "C", 445, 1.0, {}),
        (0.53, "A", 182, 0.25, {}),
        (1.51, "G", 1220, 1.0, {"stutter": 2}),
        (2.03, "H", 1350, 1.0, {}),
        (2.26, "D", 700, 1.0, {}),
        (2.50, "F", 1040, 0.3, {"bleed": True}),
        (3.00, "E", 790, 2.5, {}),
        (3.40, "C", 360, 1.0, {}),
        (3.66, "A", 140, 1.0, {"stutter": 3}),
        (4.00, "E", 875, 0.25, {}),
        (4.52, "G", 1290, 1.0, {}),
    ]),
    # 7. Float: groove, then the 808 drops out for 2.8 s: slow-mo with the giant close-up face looming in the
    #    room; the drop slams back.
    "07": dict(name="float", bars=(50, 57), edl=[
        (0.00, "G", 1155, 1.0, {}),
        (0.25, "E", 745, 1.0, {}),
        (0.53, "A", 210, 0.3, {}),
        (2.03, "D", 640, 1.0, {"tr": ("sweep", 12)}),
        (3.02, "E", 850, 1.0, {}),
        (4.02, "C", 380, 1.0, {}),
        (4.37, "A", 40, 0.35, {"calm": True, "hold": 2, "z": 1.02, "comp": 0.70, "loom": ("F", 1010, 0.4, 0.62, -0.22, 0.80)}),
        (6.00, "E", 880, 0.30, {}),
        (6.52, "H", 1360, 1.0, {}),
    ]),
    # 8. Hook 3 in triptychs: the panels stack up through the 808 break, the drop goes full frame, a second
    #    triptych builds fast mid-hook, triplet cutaways, colour cut-outs.
    "08": dict(name="triptych", bars=(63, 72), edl=[
        (0.00, "E", 760, 1.0, {"tri": [("E", 760), ("F", 1000), ("A", 200)], "tri_step": 13}),
        (1.00, "F", 1005, 0.30, {}),
        (1.33, "G", 1180, 1.0, {}),
        (1.52, "E", 830, 1.0, {}),
        (1.77, "A", 175, 0.3, {}),
        (2.77, "H", 1340, 1.0, {}),
        (2.90, "C", 455, 1.0, {}),
        (3.02, "D", 690, 0.3, {}),
        (4.02, "G", 1200, 1.0, {"tri": [("G", 1200), ("E", 880), ("F", 1060)], "tri_step": 6}),
        (4.77, "A", 90, 1.0, {}),
        (4.89, "E", 870, 1.0, {}),
        (5.02, "F", 1080, 0.3, {"bleed": True}),
        (6.02, "C", 500, 1.0, {}),
        (6.52, "G", 1260, 1.0, {"iso": 1}),
        (7.02, "D", 640, 1.0, {}),
        (7.13, "H", 1365, 1.0, {}),
        (7.26, "E", 890, 0.3, {"iso": 1}),
        (8.02, "A", 250, 1.0, {}),
    ]),
    # 9. Outro to the end of the song: the energy winds down, colour comes back from the back of the room
    #    forwards on the handshake, freeze, fade with the tail.
    "09": dict(name="outro", bars=(72, 81), fade_out=40, scan=[(7.2, 1.2, "out")], edl=[
        (0.00, "G", 1160, 1.0, {}),
        (1.02, "E", 735, 1.0, {}),
        (1.77, "A", 100, 1.0, {}),
        (2.02, "F", 950, 1.0, {}),
        (2.52, "C", 330, 1.0, {"tr": ("whip", 8)}),
        (3.77, "D", 650, 1.0, {}),
        (4.02, "E", 840, 0.3, {}),
        (5.01, "G", 1240, 1.0, {"iso": 1}),
        (6.03, "F", 1100, 1.0, {"tr": ("defocus", 10)}),
        (6.83, "H", 1318, 1.0, {"freeze": 58, "e": 0.8}),
    ]),
    # 10. Verse 2 groove: the singer cut out in colour, bar by bar, inside the ghost room.
    "10": dict(name="colour_cutout", bars=(57, 63), triplet_punch=False, edl=[
        (0.00, "A", 233, 1.0, {"iso": 1}),
        (1.02, "G", 1185, 1.0, {"tr": ("far_first", 8)}),
        (2.00, "E", 888, 0.55, {"iso": 1}),
        (3.02, "F", 1000, 1.0, {"bleed": True, "tr": ("defocus", 10)}),
        (4.01, "A", 27, 0.7, {"iso": 1}),
        (5.00, "G", 1230, 1.0, {"iso": 1}),
    ]),
}
