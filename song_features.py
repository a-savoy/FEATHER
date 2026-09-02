"""song_features.py — acoustic features and syllable segmentation for bird vocalizations.

This is FEATHER's re-implementation of the measurement half of Sound Analysis Pro
(Tchernichovski et al.), covering what you need to choose and characterise playback
stimuli: a multi-taper spectrogram with spectral derivatives, the seven SAP acoustic
features, and threshold-based segmentation into syllables and bouts.

Nothing here touches audio hardware, files, or the GUI — arrays in, tables out — so it
can be tested on any machine. See test_song_features.py for the ground-truth checks.

Defaults are tuned for ZEBRA FINCH CHICK BEGGING CALLS, not adult song:
the analysis band reaches 15 kHz because begging calls carry real energy to ~14 kHz,
where a song-tuned 8 kHz ceiling would clip them.

Reference: SAP2011 manual, chapters 3-4 (spectral analysis and song features).
Values will correlate with SAP's but not equal them; SAP's constants are not public.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

# Feature columns produced by measure(), in display order.
FEATURES = ["amplitude", "mean_frequency", "pitch", "goodness", "entropy", "fm", "am"]

FEATURE_LABELS = {
    "amplitude":      "Amplitude (dB)",
    "mean_frequency": "Mean frequency (Hz)",
    "pitch":          "Pitch (Hz)",
    "goodness":       "Goodness of pitch",
    "entropy":        "Wiener entropy",
    "fm":             "FM (degrees)",
    "am":             "AM (1/s)",
}

# Spectral defaults. Window and advance are SAP's documented values (409 samples =
# 9.27 ms at 44.1 kHz, advancing 1.36 ms for ~85% overlap); the band is ours.
SPEC_DEFAULTS = dict(
    window_ms=9.27,
    advance_ms=1.36,
    n_tapers=2,
    nw=1.5,
    f_low=800.0,
    f_high=15000.0,
    pitch_min=300.0,
    pitch_max=1800.0,
)

# Thresholds for SAP's pitch rule (manual ch. 4): use the harmonic (cepstral) estimate
# only when it is plausible, otherwise fall back to mean frequency.
PITCH_RULE_DEFAULTS = dict(
    max_cepstral_hz=1800.0,   # T1: reject cepstral pitch above this
    entropy_max=-1.0,         # T2: below this the sound is tonal enough to trust pitch
    goodness_min=0.02,        # T3: minimum harmonic periodicity to trust pitch
)

# What the sonogram SHOWS. Separate from the analysis band above on purpose: the
# display is chosen for legibility (begging calls read best over 0-7 kHz), while the
# measurements need the wider band because the calls carry energy well above it.
DISPLAY_DEFAULTS = dict(f_low=0.0, f_high=7000.0)

SEGMENT_DEFAULTS = dict(
    feature="amplitude",
    threshold=None,           # None -> peak minus threshold_below_peak_db
    threshold_below_peak_db=25.0,
    direction=">",
    smooth_points=0,
    adapt_threshold=0,        # 0 = fixed; otherwise window in frames
    min_duration_ms=15.0,
    max_duration_ms=400.0,
    min_gap_ms=8.0,
    bout_gap_ms=250.0,
    noise_highpass_hz=1000.0,  # a unit with most energy below this is not a vocalization
    noise_energy_frac=0.90,
)


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------
def _frame(x, n_win, n_adv):
    """Overlapping frames as a view: (n_frames, n_win). No copy."""
    if len(x) < n_win:
        return np.empty((0, n_win), dtype=x.dtype)
    n = 1 + (len(x) - n_win) // n_adv
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n, n_win),
                                           strides=(stride * n_adv, stride),
                                           writeable=False)


def spectral(x, fs, **params):
    """Multi-taper spectrogram, spectral derivatives, and the derivative image.

    Returns dict with:
        t          frame centre times (s)
        freqs      frequency axis within [f_low, f_high] (Hz)
        power      multi-taper power spectrum, (n_frames, n_freqs)
        time_deriv, freq_deriv   spectral derivatives, same shape
        deriv      directional derivative — the SAP-style sonogram image
        fm         per-frame FM angle (degrees), computed here since the
                   directional derivative needs it
    """
    p = dict(SPEC_DEFAULTS, **params)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    n_win = max(16, int(round(p["window_ms"] * 1e-3 * fs)))
    n_adv = max(1, int(round(p["advance_ms"] * 1e-3 * fs)))
    nfft = 1 << int(np.ceil(np.log2(n_win)))

    tapers = dpss(n_win, p["nw"], p["n_tapers"])          # (K, n_win)
    freqs_all = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs_all >= p["f_low"]) & (freqs_all <= p["f_high"])
    if not band.any():                                    # degenerate band
        band = np.ones_like(freqs_all, dtype=bool)

    frames = _frame(x, n_win, n_adv)
    if len(frames) == 0:
        z = np.empty((0, int(band.sum())))
        e = np.empty(0)
        return dict(t=e, freqs=freqs_all[band], power=z, time_deriv=z, freq_deriv=z,
                    deriv=z, fm=e, goodness=e, cep_pitch=e,
                    n_win=n_win, n_adv=n_adv, fs=fs)

    # Chunk so a long recording never materialises a huge complex array.
    out = {k: [] for k in ("power", "time_deriv", "freq_deriv", "deriv", "fm",
                           "goodness", "cep_pitch")}
    chunk = max(1, int(4e6 // max(1, nfft)))
    for i0 in range(0, len(frames), chunk):
        blk = frames[i0:i0 + chunk]
        # Two tapered FFTs. S1/S2 are the first two Slepian projections; SAP's
        # derivative estimate is built from their cross terms (Thomson 1990/93).
        S1f = np.fft.rfft(blk * tapers[0], nfft, axis=1)
        S2f = np.fft.rfft(blk * tapers[1], nfft, axis=1) if len(tapers) > 1 else S1f
        power_full = (np.abs(S1f) ** 2 + np.abs(S2f) ** 2) / len(tapers)

        # Harmonic pitch and goodness of pitch, from the cepstrum of the log power
        # spectrum. Quefrency index q is a lag in samples, so F0 = fs / q.
        ceps = np.fft.irfft(np.log(power_full + 1e-30), axis=1)
        q_lo = max(2, int(np.ceil(fs / p["pitch_max"])))
        q_hi = min(nfft // 2 - 1, int(np.floor(fs / p["pitch_min"])))
        if q_hi > q_lo:
            win = ceps[:, q_lo:q_hi + 1]
            qi = np.argmax(win, axis=1)
            out["goodness"].append(win[np.arange(len(win)), qi])
            out["cep_pitch"].append(fs / (q_lo + qi))
        else:
            out["goodness"].append(np.zeros(len(blk)))
            out["cep_pitch"].append(np.full(len(blk), np.nan))

        S1, S2 = S1f[:, band], S2f[:, band]
        power = power_full[:, band]
        td = -(S1.real * S2.real) - (S1.imag * S2.imag)
        fd = (S1.imag * S2.real) - (S1.real * S2.imag)

        # FM is the angle of the dominant modulation in each frame; the directional
        # derivative is then taken along it, which is what makes traces crisp.
        mx_t = np.abs(td).max(axis=1)
        mx_f = np.abs(fd).max(axis=1)
        fm = np.arctan2(mx_t, np.maximum(mx_f, 1e-30))
        deriv = td * np.sin(fm)[:, None] + fd * np.cos(fm)[:, None]

        out["power"].append(power)
        out["time_deriv"].append(td)
        out["freq_deriv"].append(fd)
        out["deriv"].append(deriv)
        out["fm"].append(np.degrees(fm))

    res = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    res["goodness"] = np.asarray(res["goodness"]).ravel()
    res["cep_pitch"] = np.asarray(res["cep_pitch"]).ravel()
    res["t"] = (np.arange(len(frames)) * n_adv + n_win / 2.0) / fs
    res["freqs"] = freqs_all[band]
    res["n_win"], res["n_adv"], res["fs"] = n_win, n_adv, fs
    return res


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def measure(x, fs, spec=None, **params):
    """Per-frame acoustic features. Returns (DataFrame, spectral-dict).

    The DataFrame has one row per FFT frame with columns: t + FEATURES.
    """
    p = dict(SPEC_DEFAULTS, **PITCH_RULE_DEFAULTS)
    p.update(params)
    sp = spec if spec is not None else spectral(
        x, fs, **{k: p[k] for k in SPEC_DEFAULTS})

    power, freqs = sp["power"], sp["freqs"]
    if len(power) == 0:
        return pd.DataFrame(columns=["t"] + FEATURES), sp

    eps = 1e-30
    total = power.sum(axis=1)

    # Amplitude: uncalibrated dB of in-band power.
    amplitude = 10.0 * np.log10(total + eps)

    # Wiener entropy: log(geometric mean / arithmetic mean). 0 = white noise,
    # strongly negative = pure tone. Independent of gain.
    n_f = power.shape[1]
    log_gm = np.log(power + eps).mean(axis=1)
    log_am = np.log(np.maximum(total / n_f, eps))
    entropy = log_gm - log_am

    # Mean frequency: centroid of derivative power (manual ch. 4).
    dpow = np.abs(sp["time_deriv"]) + np.abs(sp["freq_deriv"])
    dsum = np.maximum(dpow.sum(axis=1), eps)
    mean_frequency = (dpow * freqs[None, :]).sum(axis=1) / dsum

    # AM: signed time-derivative power — positive at onsets, negative at offsets.
    am = sp["time_deriv"].sum(axis=1) / np.maximum(total, eps)

    fm = sp["fm"]
    goodness, cep_pitch = sp["goodness"], sp["cep_pitch"]

    # SAP's pitch rule: trust the harmonic estimate only where it is plausible,
    # otherwise report mean frequency.
    use_mean = (~np.isfinite(cep_pitch) | (cep_pitch > p["max_cepstral_hz"]) |
                ((entropy > p["entropy_max"]) & (goodness < p["goodness_min"])))
    pitch = np.where(use_mean, mean_frequency, cep_pitch)

    df = pd.DataFrame({
        "t": sp["t"],
        "amplitude": amplitude,
        "mean_frequency": mean_frequency,
        "pitch": pitch,
        "goodness": goodness,
        "entropy": entropy,
        "fm": fm,
        "am": am,
    })
    return df, sp


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def _smooth(v, n):
    if not n or n < 2:
        return v
    k = np.ones(int(n)) / float(int(n))
    return np.convolve(v, k, mode="same")


def _runs(mask):
    """Start/stop indices (half-open) of True runs."""
    if not len(mask):
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask))
    return list(zip(starts, stops))


def segment(feat, spec=None, **params):
    """Segment per-frame features into syllables, then group them into bouts.

    Mirrors SAP's segmentation panel: threshold a chosen feature, optionally smoothed
    and optionally adaptive, then apply duration and gap constraints and a high-pass
    noise test that rejects units whose energy is mostly below noise_highpass_hz.

    Returns a DataFrame with one row per syllable.
    """
    p = dict(SEGMENT_DEFAULTS, **params)
    if len(feat) == 0:
        return pd.DataFrame(columns=["syllable", "bout", "start_s", "end_s", "duration_ms"])

    t = feat["t"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.001
    v = _smooth(feat[p["feature"]].to_numpy(float), p["smooth_points"])

    thr = p["threshold"]
    if thr is None:
        thr = np.nanmax(v) - p["threshold_below_peak_db"]
    if p["adapt_threshold"]:
        # SAP's adaptive threshold: it rides the smoothed feature so a quiet passage
        # is not lost to a level set on a loud one.
        base = _smooth(v, int(p["adapt_threshold"]))
        thr = base - (np.nanmean(base) - thr)
    mask = (v > thr) if p["direction"] == ">" else (v < thr)

    runs = _runs(np.asarray(mask))

    # Merge units separated by less than min_gap.
    min_gap = p["min_gap_ms"] * 1e-3
    merged = []
    for a, b in runs:
        if merged and (t[a] - t[merged[-1][1] - 1]) < min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    min_d, max_d = p["min_duration_ms"] * 1e-3, p["max_duration_ms"] * 1e-3
    rows = []
    for a, b in merged:
        dur = (b - a) * dt
        if dur < min_d or dur > max_d:
            continue
        if spec is not None and p["noise_highpass_hz"]:
            # High-pass noise detector: mostly-low-frequency energy is the cage,
            # not the bird.
            pw = spec["power"][a:b]
            lo = pw[:, spec["freqs"] < p["noise_highpass_hz"]].sum()
            if pw.sum() > 0 and lo / pw.sum() > p["noise_energy_frac"]:
                continue
        row = dict(start_s=float(t[a]), end_s=float(t[a] + dur), duration_ms=dur * 1e3)
        seg = feat.iloc[a:b]
        for f in FEATURES:
            row[f"{f}_mean"] = float(seg[f].mean())
            row[f"{f}_var"] = float(seg[f].var())
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["syllable", "bout", "start_s", "end_s", "duration_ms"])

    # Bouts: syllables closer together than bout_gap belong to the same bout.
    gaps = out["start_s"].to_numpy()[1:] - out["end_s"].to_numpy()[:-1]
    bout = np.concatenate([[0], np.cumsum(gaps > p["bout_gap_ms"] * 1e-3)])
    out.insert(0, "bout", bout.astype(int))
    out.insert(0, "syllable", np.arange(1, len(out) + 1))
    out["cluster_id"] = -1        # reserved: syllable types, not computed yet
    return out


def bouts(syl):
    """Collapse a syllable table into one row per bout."""
    if syl is None or len(syl) == 0:
        return pd.DataFrame(columns=["bout", "start_s", "end_s", "duration_ms", "n_syllables"])
    g = syl.groupby("bout")
    out = pd.DataFrame({
        "bout": g.size().index,
        "start_s": g["start_s"].min().to_numpy(),
        "end_s": g["end_s"].max().to_numpy(),
        "n_syllables": g.size().to_numpy(),
    })
    out["duration_ms"] = (out["end_s"] - out["start_s"]) * 1e3
    return out


def analyze(x, fs, spec_params=None, segment_params=None):
    """Convenience: waveform in, (features, syllables, spectral) out."""
    feat, sp = measure(x, fs, **(spec_params or {}))
    syl = segment(feat, spec=sp, **(segment_params or {}))
    return feat, syl, sp
