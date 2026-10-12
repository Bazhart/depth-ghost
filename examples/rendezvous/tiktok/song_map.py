#!/usr/bin/env python3
"""Map the song: beat grid phase at 140 BPM, per-bar features, and 8-bar repetition (to find the hook)."""
import subprocess, sys
import numpy as np
SR, BPM = 22050, 140.0
BEAT = 60 / BPM
path = sys.argv[1]
x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                 capture_output=True, check=True).stdout, np.float32)
N_FFT, HOP = 4096, 256
win = np.hanning(N_FFT).astype(np.float32)
nfr = 1 + (len(x) - N_FFT) // HOP
S = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:nfr] * win, axis=1)).astype(np.float32)
freqs = np.fft.rfftfreq(N_FFT, 1 / SR); tfr = (np.arange(nfr) * HOP + N_FFT / 2) / SR
# onset strength (log spectral flux) -> grid phase
L = np.log1p(100 * S)
flux = np.maximum(np.diff(L, axis=0), 0).sum(1); flux = np.concatenate([[0], flux])
best = None
for ph in np.arange(0, BEAT, 0.002):
    k = np.round((np.arange(ph, tfr[-1], BEAT / 2) - tfr[0]) * SR / HOP).astype(int)  # eighth notes
    k = k[(k >= 0) & (k < nfr)]
    sc = flux[k].mean()
    if best is None or sc > best[0]: best = (sc, ph)
t0 = best[1]
# downbeat: which of the 4 beat phases has most sub-bass onsets
sub = S[:, (freqs > 30) & (freqs < 110)].sum(1); subf = np.maximum(np.diff(np.log1p(sub)), 0); subf = np.concatenate([[0], subf])
scores = []
for b in range(4):
    ts = np.arange(t0 + b * BEAT, tfr[-1], 4 * BEAT); k = np.round((ts - tfr[0]) * SR / HOP).astype(int); k = k[k < nfr]
    scores.append(subf[k].mean())
print(f"beat phase t0 = {t0:.3f}s ; sub-onset by beat-in-bar offset: {np.round(scores, 3)}")
bar0 = t0 + int(np.argmax(scores)) * BEAT
while bar0 - 4 * BEAT > -0.05: bar0 -= 4 * BEAT
BAR = 4 * BEAT; nb = int((len(x) / SR - bar0) // BAR)
print(f"bar grid starts {bar0:.3f}s, bar = {BAR:.4f}s, {nb} bars")
# per-bar features
def band(lo, hi): return S[:, (freqs >= lo) & (freqs < hi)]
pc = (np.round(12 * np.log2(np.maximum(freqs, 1) / 440)) % 12).astype(int)
chroma_mask = (freqs > 80) & (freqs < 4000)
C = np.stack([S[:, chroma_mask & (pc == p)].sum(1) for p in range(12)], 1)
edges = np.geomspace(40, 10000, 25)
E = np.log1p(np.stack([S[:, (freqs >= edges[j]) & (freqs < edges[j + 1])].sum(1) for j in range(24)], 1))
rows, feats = [], []
for b in range(nb):
    a, z = bar0 + b * BAR, bar0 + (b + 1) * BAR
    k = (tfr >= a) & (tfr < z)
    rms = 20 * np.log10(np.sqrt((x[int(a * SR):int(z * SR)] ** 2).mean()) + 1e-9)
    s808 = 20 * np.log10(band(30, 90)[k].sum(1).mean() + 1e-9)
    voc = 20 * np.log10(band(300, 3000)[k].sum(1).mean() + 1e-9)
    hat = 20 * np.log10(band(7000, 11000)[k].sum(1).mean() + 1e-9)
    rows.append((b, a, rms, s808, voc, hat))
    # beat-synchronous chroma + envelope for this bar (4 beats x features)
    fb = []
    for q in range(8):
        kk = (tfr >= a + q * BAR / 8) & (tfr < a + (q + 1) * BAR / 8)
        c = C[kk].mean(0); c = c / (np.linalg.norm(c) + 1e-9)
        e = E[kk].mean(0); e = (e - e.mean()) / (e.std() + 1e-9) / np.sqrt(24)
        fb.append(np.concatenate([c, e]))
    feats.append(np.concatenate(fb) / np.sqrt(16))
F = np.array(feats)
SSM = F @ F.T
print("\nbar  start   rms   808   mid   hat")
for b, a, rms, s808, voc, hat in rows:
    print(f"{b:3d} {a:7.2f} {rms:6.1f} {s808:5.1f} {voc:5.1f} {hat:5.1f}")
# 8-bar block similarity
blk = 8
M = np.array([[np.mean([SSM[i + q, j + q] for q in range(blk)]) for j in range(nb - blk + 1)] for i in range(nb - blk + 1)])
print("\nmost repeated 8-bar blocks (start bar, time, best match elsewhere >= 8 bars away):")
cand = []
for i in range(nb - blk + 1):
    others = [(M[i, j], j) for j in range(nb - blk + 1) if abs(i - j) >= 8]
    others.sort(reverse=True)
    cand.append((sum(v for v, _ in others[:2]), i, others[:3]))
cand.sort(reverse=True)
for sc, i, o in cand[:14]:
    print(f"  bar {i:3d} @ {bar0 + i * BAR:6.2f}s  score {sc:.3f}  matches " + ", ".join(f"bar {j} ({bar0 + j * BAR:.2f}s, {v:.3f})" for v, j in o))
np.save(sys.argv[2], dict(bar0=bar0, BAR=BAR, nb=nb, rows=rows, M=M), allow_pickle=True)
