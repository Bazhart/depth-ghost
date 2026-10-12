#!/usr/bin/env python3
"""Musical events over the whole song, saved to song_events.json:
joint  808 + open-hat hits. Detected on the earlier mix (same notes; there the detector reproduces the 74 hand-
       checked hits of the 47 s edit exactly), then each kept only if the new mix has a hat onset within 40 ms.
groups three evenly spaced joint hits (the "dun dun dun").
breaks stretches where the 808 drops out (new mix).  drops: the 808's return after a break.
hits   strong transients of the new mix (drive the glow pulse)."""
import json, subprocess, sys
import numpy as np
NEW, OLD = sys.argv[1], sys.argv[2]
SR, HOP, NFFT = 44100, 256, 2048; FE = SR / HOP


def spec(path):
    x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP] * np.hanning(NFFT)
    return x, np.abs(np.fft.rfft(fr, axis=1)).astype(np.float32)


F = np.fft.rfftfreq(NFFT, 1 / SR)


def band_env(S, lo, hi, norm_n):
    m = np.log1p(S[:, (F >= lo) & (F < hi)] * 30).mean(1)
    on = np.maximum(np.diff(m, prepend=m[0]), 0); on = np.convolve(on, np.ones(3) / 3, "same")
    return m, on / (np.percentile(on[:norm_n], 99.5) + 1e-9)


def peaks(on, th, gap):
    return np.array([i for i in range(1, len(on) - 1) if on[i] > th and on[i] == on[max(0, i - gap):i + gap + 1].max()])


xo, So = spec(OLD); xn, Sn = spec(NEW)
nn = int(46.9 * FE)
_, so = band_env(So, 25, 110, nn); _, ho = band_env(So, 6000, 14000, nn)
ps, ph = peaks(so, 0.35, int(0.08 * FE)), peaks(ho, 0.35, int(0.06 * FE))
cand = [(i, so[i] + ho[ph[np.argmin(np.abs(ph - i))]]) for i in ps if len(ph) and np.abs(ph - i).min() < 0.035 * FE]
joint = []                                           # merge detections closer than 60 ms (keep the strongest)
for i, s in cand:
    t = i / FE
    if joint and t - joint[-1][0] < 0.06:
        if s > joint[-1][1]: joint[-1] = (t, s)
    else:
        joint.append((t, s))
_, hn = band_env(Sn, 6000, 14000, nn); phn = peaks(hn, 0.25, int(0.04 * FE)) / FE
joint = [(t + 0.0064, s) for t, s in joint if len(phn) and np.abs(phn - t).min() < 0.04]   # the new mix sits 6.4 ms later
bt = np.array([t for t, _ in joint])
groups, k = [], 0
while k + 2 < len(bt):
    d1, d2 = bt[k + 1] - bt[k], bt[k + 2] - bt[k + 1]
    if 0.09 < d1 < 0.5 and 0.09 < d2 < 0.5 and abs(d1 - d2) < 0.3 * max(d1, d2):
        groups.append([round(float(v), 3) for v in bt[k:k + 3]]); k += 3
    else:
        k += 1
BAR = 12 / 7; BAR0 = -0.052
sub_lvl = np.log1p(Sn[:, (F >= 25) & (F < 110)] * 30).mean(1)
sl = np.convolve(sub_lvl, np.ones(9) / 9, "same"); ref = np.percentile(sl, 60); floor = np.percentile(sl, 2)
eighth = BAR / 8; nseg = int((len(xn) / SR - BAR0) / eighth) - 1
lv = np.array([sl[max(0, int((BAR0 + q * eighth) * FE)):int((BAR0 + (q + 1) * eighth) * FE)].mean() for q in range(nseg)])
low = lv < ref - 0.35 * (ref - floor)
breaks, q = [], 0
while q < nseg:
    if low[q]:
        r = q
        while r + 1 < nseg and low[r + 1]: r += 1
        if r - q + 1 >= 2: breaks.append([round(BAR0 + q * eighth, 3), round(BAR0 + (r + 1) * eighth, 3)])
        q = r + 1
    else:
        q += 1
drops = [b for _, b in breaks if b < len(xn) / SR - 2]
x2 = xn[::2]; sr2 = 22050
fr2 = np.lib.stride_tricks.sliding_window_view(x2, 1024)[::256] * np.hanning(1024)
m2 = np.log1p(np.abs(np.fft.rfft(fr2, axis=1)) * 10)
env = np.maximum(np.diff(m2, axis=0), 0).sum(1); t2 = (np.arange(len(env)) + 1) * 256 / sr2
NF = int(len(xn) / SR * 30); per = np.zeros(NF)
np.maximum.at(per, np.minimum((t2 * 30).astype(int), NF - 1), env)
hits, last = [], -99
for i in range(1, NF - 1):
    w = per[max(0, i - 450):i + 450]
    z = (per[i] - np.median(w)) / (w.std() + 1e-9)
    if z > 2.2 and per[i] == per[max(0, i - 6):i + 7].max() and i - last >= 12:
        hits.append(round(i / 30, 3)); last = i
json.dump(dict(bar0=BAR0, bar=BAR, beat=BAR / 4, joint=[round(float(t), 3) for t in bt], groups=groups, breaks=breaks,
               drops=drops, hits=hits, dur=len(xn) / SR), open("song_events.json", "w"), indent=1)
print(f"joint hits {len(bt)} | groups {len(groups)} | breaks {len(breaks)} | transients {len(hits)}")
JT = [0.84, 1.70, 2.54, 3.41, 3.62, 4.06, 4.27, 4.86, 5.12, 5.54, 5.97, 6.40, 6.83, 7.05, 7.70, 8.55, 9.41, 10.27, 10.91, 11.12, 11.64, 11.98, 12.41, 13.18, 13.68, 14.25, 14.56, 15.41, 16.25, 17.12, 17.76, 17.98, 18.62, 18.83, 20.12, 20.33, 20.54, 22.25, 22.88, 23.10, 23.98, 24.61, 24.83, 25.47, 25.69, 26.53, 26.97, 27.19, 27.41, 28.27, 28.70, 31.23, 31.47, 31.70, 31.89, 32.13, 32.33, 32.55, 35.02, 35.77, 35.98, 37.12, 37.69, 37.87, 38.02, 38.34, 38.55, 39.19, 39.40, 41.11, 41.98, 42.84, 45.05, 46.69]
print(f"hand-checked 0-47 s hits kept: {sum(np.abs(bt - t).min() < 0.04 for t in JT)}/{len(JT)}; extra in 0-47 s: {sum(np.abs(np.array(JT) - t).min() >= 0.04 for t in bt[bt < 46.9])}")
print("groups:", "  ".join(f"{g[0]:.2f}" for g in groups)); print("breaks:", breaks)
for name, a, b in (("intro", 0, 13.66), ("hook 1", 13.66, 27.38), ("verse 1", 27.38, 54.80), ("hook 2", 54.80, 68.52),
                   ("verse 2", 68.52, 109.66), ("hook 3", 109.66, 123.38), ("outro", 123.38, 141.4)):
    js = bt[(bt >= a) & (bt < b)]
    print(f"\n{name} {a:.2f}-{b:.2f}: {len(js)} joint hits (bar-relative: bar.beat)")
    print("  " + " ".join(f"{(t - BAR0) / BAR:.2f}" for t in js))
