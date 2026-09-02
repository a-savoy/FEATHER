"""make_test_data.py — generate a FAKE Open Ephys recording to test the pipeline.

It writes a recording that looks like a real session: 128 neural channels with
simulated neurons (a few of which fire more after each sound), plus the ADC1 timing
gate and the ADC2 audio copy, matched to the H5-D probe geometry. You can then run
the notebook's Steps 7-9 (find onsets -> Kilosort -> PSTH) on it.

Everything is synthetic; delete the test_data folder when done.
"""
import os
import json
import csv
import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from probeinterface import read_probeinterface

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe_H5D.json")
STIM = os.path.join(HERE, "stimuli")
REC = os.path.join(HERE, "test_data", "SYNTHETIC", "Record Node 122", "experiment1", "recording1")
FOLDER = "synthetic_neural"

fs = 30000
n_neural = 128
adc_names = ["ADC1", "ADC2", "ADC3", "ADC4", "ADC5", "ADC6", "ADC7", "ADC8"]
n_adc = len(adc_names)
n_ch = n_neural + n_adc

n_events = 15
lead_in = 4.0
period = 5.7                       # seconds between sound onsets
dur_s = lead_in + n_events * period + 3.0
n_samp = int(dur_s * fs)
rng = np.random.default_rng(7)

# ---- probe geometry + channel map ----
probe = read_probeinterface(PROBE).probes[0]
pos = probe.contact_positions                                   # (128, 2) x,y in um
chanmap = np.asarray(probe.device_channel_indices, dtype=int)   # contact i -> .dat channel

# ---- stimuli (for the ADC2 audio copy), resampled to fs ----
stim_files = sorted(f for f in os.listdir(STIM) if f.lower().endswith(".wav"))
stims = {}
for f in stim_files:
    x, sr = sf.read(os.path.join(STIM, f), dtype="float64", always_2d=True)
    x = x[:, 0]
    if int(sr) != fs:
        g = gcd(fs, int(sr))
        x = resample_poly(x, fs // g, int(sr) // g)
    stims[f] = x

onsets_s = [lead_in + i * period for i in range(n_events)]
seq = [stim_files[int(rng.integers(len(stim_files)))] for _ in range(n_events)]

# ---- master buffer (samples x channels), int16 ----
master = np.zeros((n_samp, n_ch), dtype=np.int16)

# neural background noise, in 10 s blocks to keep memory low
NOISE_SD = 45
blk = fs * 10
for t0 in range(0, n_samp, blk):
    t1 = min(n_samp, t0 + blk)
    master[t0:t1, :n_neural] = rng.normal(0, NOISE_SD, size=(t1 - t0, n_neural)).astype(np.int16)

# spike template (~2 ms, biphasic; peak scaled to 1)
tw = np.arange(-20, 40)
template = -np.exp(-0.5 * (tw / 3.0) ** 2) + 0.4 * np.exp(-0.5 * ((tw - 12) / 6.0) ** 2)
template = template / np.max(np.abs(template))

# ---- neurons ----
n_units, n_aud = 10, 3
lam = 30.0                                     # um: amplitude space constant
units = []
for j in range(n_units):
    ci = int(rng.integers(n_neural))
    d = np.sqrt(((pos - pos[ci]) ** 2).sum(1))
    amp = float(rng.uniform(320, 650))
    units.append(dict(near=ci, chan_amp=amp * np.exp(-d / lam),
                      rate=float(rng.uniform(3, 8)), auditory=(j < n_aud)))


def stamp(s0, chan_amp):
    if s0 - 20 < 0 or s0 + 40 > n_samp:
        return
    for ci in np.where(chan_amp > 8)[0]:
        ch = chanmap[ci]
        seg = master[s0 - 20:s0 + 40, ch].astype(np.int32) + (chan_amp[ci] * template).astype(np.int32)
        master[s0 - 20:s0 + 40, ch] = np.clip(seg, -32000, 32000).astype(np.int16)


gt_counts = []
for u in units:
    times = list(rng.uniform(0.05, dur_s - 0.05, size=rng.poisson(u["rate"] * dur_s)))
    if u["auditory"]:
        for o in onsets_s:
            for lat in rng.normal(0.12, 0.03, size=rng.poisson(4)):   # ~120 ms response
                tt = o + max(0.03, float(lat))
                if 0 < tt < dur_s:
                    times.append(tt)
    times = np.sort(times)
    for t in times:
        stamp(int(t * fs), u["chan_amp"])
    gt_counts.append(len(times))

# ---- ADC channels: gate (ADC1), audio (ADC2), rest low noise ----
gate = np.zeros(n_samp, dtype=np.int16)
audio = np.zeros(n_samp, dtype=np.int16)
for o, f in zip(onsets_s, seq):
    s0 = int(o * fs)
    stim = stims[f]
    e = min(n_samp, s0 + len(stim))
    gate[s0:e] = 6000
    audio[s0:e] = (np.clip(stim[:e - s0], -1, 1) * 8000).astype(np.int16)
master[:, n_neural + 0] = gate
master[:, n_neural + 1] = audio
for c in range(2, n_adc):
    master[:, n_neural + c] = rng.normal(0, 20, size=n_samp).astype(np.int16)

# ---- write continuous.dat + structure.oebin ----
os.makedirs(os.path.join(REC, "continuous", FOLDER), exist_ok=True)
master.tofile(os.path.join(REC, "continuous", FOLDER, "continuous.dat"))

channels = [dict(channel_name=f"CH{i + 1}", bit_volts=0.195) for i in range(n_neural)]
channels += [dict(channel_name=nm, bit_volts=0.00015258789) for nm in adc_names]
oebin = dict(GUI_version="synthetic-test",
             continuous=[dict(folder_name=FOLDER + "/", sample_rate=fs,
                              source_processor_name="Synthetic", source_processor_id=100,
                              num_channels=n_ch, channels=channels)],
             events=[], spikes=[])
json.dump(oebin, open(os.path.join(REC, "structure.oebin"), "w"), indent=2)

# ---- ground truth ----
with open(os.path.join(REC, "GROUND_TRUTH.csv"), "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["sound_number", "onset_seconds", "sound_file"])
    for i, (o, f) in enumerate(zip(onsets_s, seq)):
        w.writerow([i + 1, round(o, 4), f])

sizemb = os.path.getsize(os.path.join(REC, "continuous", FOLDER, "continuous.dat")) / 1e6
print(f"wrote fake recording ({dur_s:.0f}s, {n_ch} channels, {sizemb:.0f} MB):")
print(" ", REC)
print(f"{n_units} simulated neurons ({n_aud} respond to sound ~120 ms after onset); "
      f"spikes/neuron: {gt_counts}")
print(f"{n_events} sound events; ground truth in GROUND_TRUTH.csv")
