"""playback_tools.py

Helper functions for building and rendering audio playback experiments.
Imported by playback_experiment.ipynb. All the experiment logic lives here so
the notebook itself stays short and readable.

Pipeline in one line:
    scan your WAVs -> build a randomized trial schedule -> render ONE master
    WAV (stimuli + silence gaps) that Bonsai plays start-to-finish.

Dependencies: numpy, pandas, soundfile, matplotlib.
    pip install numpy pandas soundfile matplotlib
"""
from __future__ import annotations

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf


# ---------------------------------------------------------------------------
# 1. (Optional) make demo stimuli so you can test the whole pipeline with
#    nothing of your own yet.
# ---------------------------------------------------------------------------
def make_demo_stimuli(folder, sample_rate=44100, seed=0):
    """Write three short synthetic 'song-like' WAVs into `folder` so you can
    run the entire pipeline before you have real stimuli. Returns the paths.

    Delete these and drop in your own WAVs whenever you're ready.
    """
    os.makedirs(folder, exist_ok=True)
    specs = {
        "demo_A.wav": [(1200, 0.18), (2400, 0.12), (1800, 0.20)],
        "demo_B.wav": [(800, 0.25), (1600, 0.15)],
        "demo_C.wav": [(3000, 0.10), (2000, 0.10), (1000, 0.22), (2500, 0.14)],
    }
    paths = []
    for name, syllables in specs.items():
        chunks = []
        for freq, dur in syllables:
            t = np.arange(int(dur * sample_rate)) / sample_rate
            tone = 0.6 * np.sin(2 * np.pi * freq * t)
            ramp = max(1, int(0.01 * sample_rate))          # 10 ms fade in/out
            env = np.ones_like(tone)
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
            chunks.append((tone * env).astype(np.float32))
            chunks.append(np.zeros(int(0.03 * sample_rate), np.float32))  # gap
        sig = np.concatenate(chunks).astype(np.float32)
        path = os.path.join(folder, name)
        sf.write(path, sig, sample_rate, subtype="PCM_16")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 2. Scan a folder of WAVs and report what's there.
# ---------------------------------------------------------------------------
def scan_stimuli(folder, extensions=(".wav",)):
    """Read every WAV in `folder` and return a DataFrame:
        file, duration_s, sample_rate, channels, n_samples
    Raises if the folder has no matching files.
    """
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(extensions))
    if not files:
        raise FileNotFoundError(
            f"No WAV files found in {folder!r}. Put your stimuli there, "
            "or run the 'make demo stimuli' cell.")
    rows = []
    for f in files:
        info = sf.info(os.path.join(folder, f))
        rows.append(dict(file=f,
                         duration_s=round(info.duration, 4),
                         sample_rate=info.samplerate,
                         channels=info.channels,
                         n_samples=info.frames))
    return pd.DataFrame(rows)


def check_uniform_sample_rate(stim_table):
    """Return the common sample rate, or raise if the WAVs disagree.
    (The master file has one sample rate, so all stimuli must match.)"""
    rates = sorted(set(stim_table["sample_rate"]))
    if len(rates) != 1:
        raise ValueError(
            f"Your stimuli have different sample rates: {rates}. "
            "Re-export them all at one rate (44100 or 48000 Hz is typical).")
    return int(rates[0])


# ---------------------------------------------------------------------------
# 3. Build the randomized trial schedule.
# ---------------------------------------------------------------------------
def build_schedule(stim_table, repeats=10, randomize_order=True,
                   avoid_immediate_repeat=True,
                   gap_mode="random", gap_fixed=3.0,
                   gap_min=2.0, gap_max=5.0, gap_set=None,
                   lead_in=2.0, tail=2.0, seed=None):
    """Build the full playback list, one row per stimulus presentation.

    Parameters
    ----------
    stim_table : DataFrame from scan_stimuli().
    repeats : int, or dict {filename: n}. How many times each stimulus plays.
    randomize_order : True = shuffle the order; False = fixed cycle A,B,C,A,B,C...
    avoid_immediate_repeat : if shuffling, never play the same stimulus twice
        in a row.
    gap_mode : "random" (uniform in [gap_min, gap_max]), "fixed" (gap_fixed),
        or "set" (each gap drawn uniformly at random from the list `gap_set`,
        e.g. [5, 10, 15, 20, 25]).
    gap_fixed : seconds, used when gap_mode == "fixed".
    gap_min, gap_max : seconds, used when gap_mode == "random".
    gap_set : list of seconds to draw from, used when gap_mode == "set".
    lead_in : silence before the first stimulus (seconds).
    tail : silence after the last stimulus (seconds).
    seed : int for reproducible order + gaps, or None for a fresh random draw.

    Returns
    -------
    DataFrame with columns:
        trial, file, duration_s, gap_after_s, onset_s, offset_s
    onset_s / offset_s are seconds from the start of the master WAV.
    """
    rng = random.Random(seed)
    dur = dict(zip(stim_table["file"], stim_table["duration_s"]))
    files = list(stim_table["file"])

    # pool of presentations
    if isinstance(repeats, dict):
        pool = []
        for f in files:
            pool += [f] * int(repeats.get(f, 0))
    else:
        pool = files * int(repeats)
    if not pool:
        raise ValueError("Schedule is empty - check `repeats`.")

    # order
    if randomize_order:
        order = (_shuffle_no_adjacent(pool, rng) if avoid_immediate_repeat
                 else _shuffle(pool, rng))
    else:
        order = pool  # fixed cycle A,B,C,A,B,C... (or blocked if repeats is a dict)

    # gaps + timeline
    rows, onset = [], float(lead_in)
    n = len(order)
    if gap_mode == "set" and not gap_set:
        raise ValueError("gap_mode='set' requires a non-empty gap_set list, "
                         "e.g. gap_set=[5, 10, 15, 20, 25].")
    for i, f in enumerate(order):
        d = float(dur[f])
        if gap_mode == "fixed":
            gap = float(gap_fixed)
        elif gap_mode == "set":
            gap = float(rng.choice(list(gap_set)))
        else:
            gap = round(rng.uniform(gap_min, gap_max), 3)
        if i == n - 1:
            gap = float(tail)          # silence after the last stimulus
        rows.append(dict(trial=i + 1, file=f, duration_s=round(d, 4),
                         gap_after_s=round(gap, 3),
                         onset_s=round(onset, 4),
                         offset_s=round(onset + d, 4)))
        onset = onset + d + gap
    return pd.DataFrame(rows)


def _shuffle(pool, rng):
    out = pool[:]
    rng.shuffle(out)
    return out


def _shuffle_no_adjacent(pool, rng, max_attempts=2000):
    """Shuffle so no two identical items are adjacent."""
    counts = Counter(pool)
    n = len(pool)
    if max(counts.values()) > (n + 1) // 2:
        raise ValueError(
            "Cannot avoid immediate repeats: one stimulus is repeated too many "
            "times relative to the others. Add more stimulus types, reduce its "
            "repeats, or set avoid_immediate_repeat=False.")
    # cheap path: random retries
    for _ in range(max_attempts):
        cand = _shuffle(pool, rng)
        if all(cand[i] != cand[i + 1] for i in range(n - 1)):
            return cand
    # guaranteed fallback: greedy, always pick the most-remaining != last
    remaining, out, last = dict(counts), [], None
    for _ in range(n):
        choices = [k for k, v in remaining.items() if v > 0 and k != last] \
                  or [k for k, v in remaining.items() if v > 0]
        top = max(remaining[k] for k in choices)
        pick = rng.choice([k for k in choices if remaining[k] == top])
        out.append(pick)
        remaining[pick] -= 1
        last = pick
    return out


def summarize(schedule, sample_rate=None):
    """Print a human-readable summary of a schedule."""
    total_s = float(schedule["offset_s"].iloc[-1] +
                    schedule["gap_after_s"].iloc[-1])
    mm, ss = divmod(int(round(total_s)), 60)
    print(f"Trials:            {len(schedule)}")
    print("Per stimulus:      " +
          ", ".join(f"{k}x{v}" for k, v in
                    schedule['file'].value_counts().sort_index().items()))
    gaps = schedule["gap_after_s"].iloc[:-1]
    if len(gaps):
        print(f"Gaps (s):          {gaps.min():.2f} - {gaps.max():.2f} "
              f"(mean {gaps.mean():.2f})")
    print(f"Total duration:    {mm}:{ss:02d} (mm:ss) = {total_s:.1f} s")
    if sample_rate:
        print(f"Master WAV length: {int(round(total_s * sample_rate)):,} samples "
              f"@ {sample_rate} Hz")


# ---------------------------------------------------------------------------
# 4. Render the master playback WAV from the schedule.
# ---------------------------------------------------------------------------
def render_master_wav(schedule, stimuli_dir, out_path, sample_rate,
                      channels=1, amplitude=1.0, subtype="PCM_16"):
    """Concatenate stimuli + silence gaps into ONE WAV that Bonsai plays
    start-to-finish. Stimulus onsets land exactly at schedule['onset_s'].

    channels : 1 = mono (stereo stimuli are down-mixed); 2 = duplicate to stereo.
    amplitude : global scale (1.0 = as-is). Kept <= 1.0 to avoid clipping.
    Returns (out_path, total_seconds).
    """
    total_s = float(schedule["offset_s"].iloc[-1] +
                    schedule["gap_after_s"].iloc[-1])
    total_samples = int(round(total_s * sample_rate))
    master = np.zeros((total_samples, channels), dtype=np.float32)

    cache = {}
    for _, row in schedule.iterrows():
        f = row["file"]
        if f not in cache:
            data, sr = sf.read(os.path.join(stimuli_dir, f),
                               dtype="float32", always_2d=True)
            if sr != sample_rate:
                raise ValueError(
                    f"{f} is {sr} Hz but the master is {sample_rate} Hz. "
                    "Re-export all stimuli at the same rate.")
            if data.shape[1] != channels:            # fix channel count
                data = (data.mean(axis=1, keepdims=True) if channels == 1
                        else np.repeat(data[:, :1], channels, axis=1))
            cache[f] = data
        clip = cache[f] * float(amplitude)
        start = int(round(row["onset_s"] * sample_rate))
        end = min(start + clip.shape[0], total_samples)
        master[start:end, :] += clip[:end - start, :]

    peak = float(np.max(np.abs(master))) if master.size else 0.0
    if peak > 1.0:                                    # safety limiter
        master = master / peak * 0.99
        print(f"[note] Peak exceeded 1.0; scaled down by {peak:.2f}x to avoid clipping.")
    sf.write(out_path, master, sample_rate, subtype=subtype)
    return out_path, total_s


# ---------------------------------------------------------------------------
# 5. Preview the schedule as a timeline.
# ---------------------------------------------------------------------------
def plot_timeline(schedule, max_seconds=None):
    """Draw the playback timeline: one colored bar per stimulus, gaps as space."""
    import matplotlib.pyplot as plt

    stims = sorted(schedule["file"].unique())
    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i % 10) for i, s in enumerate(stims)}

    fig, ax = plt.subplots(figsize=(11, 2.4))
    for _, r in schedule.iterrows():
        ax.barh(0, r["offset_s"] - r["onset_s"], left=r["onset_s"], height=0.6,
                color=color[r["file"]], edgecolor="none")
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Playback timeline - {len(schedule)} stimuli")
    if max_seconds:
        ax.set_xlim(0, max_seconds)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[s]) for s in stims]
    ax.legend(handles, stims, ncol=len(stims), loc="upper center",
              bbox_to_anchor=(0.5, -0.35), frameon=False)
    fig.tight_layout()
    return fig
