"""song_generate.py — synthetic begging calls from the trained VAE.

FEATHER's side of the zebra finch begging-call generator. The model itself lives
unmodified in callgen/ (see callgen/SOURCE.md); this module is the thin layer that
FEATHER needs on top of it:

  * describes every control as a slider with BOTH its species-typical range and the
    wider range the sliders allow, because generating deliberately supernormal stimuli
    is the point of the tool, not an accident;
  * renders to audio at 44.1 kHz, the rate FEATHER's stimulus folders already use;
  * saves a generated call as a stimulus with its full parameter set recorded, so a
    synthetic stimulus is exactly as reproducible as a recorded one is traceable.

A generated stimulus is a WAV like any other. It plays through the Harp card during
ephys, or through the computer's speakers for behavioural work, with no special case
anywhere downstream.
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
CALLGEN = os.path.join(HERE, "callgen")
SR = 44100

_ENGINE = {"mod": None, "error": None}


def available():
    """Is the generator installed? (Returns (ok, message).)"""
    need = ["generator_engine.py", "audio_utils.py", "train_vae.py",
            "vae_ckpt.pt", "vae_norm_stats.npz", "latent_directions.npz"]
    missing = [f for f in need if not os.path.exists(os.path.join(CALLGEN, f))]
    if missing:
        return False, ("The call generator is not installed. Missing from callgen/: "
                       + ", ".join(missing) + ".\nSee callgen/SOURCE.md.")
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "PyTorch is not installed in this environment, so the generator cannot run."
    return True, ""


def engine():
    """Import the vendored engine, once."""
    if _ENGINE["mod"] is not None:
        return _ENGINE["mod"]
    if _ENGINE["error"]:
        raise RuntimeError(_ENGINE["error"])
    ok, msg = available()
    if not ok:
        _ENGINE["error"] = msg
        raise RuntimeError(msg)
    if CALLGEN not in sys.path:
        sys.path.insert(0, CALLGEN)      # the engine uses absolute imports
    import generator_engine as ge
    _ENGINE["mod"] = ge
    return ge


def empirical_ranges():
    path = os.path.join(CALLGEN, "empirical_ranges.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"ranges": {}, "counts": {}}


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
# key, label, slider min, slider max, step, typical (min, max) or None, help.
# Slider ranges deliberately extend past the species-typical range: values outside it
# are supernormal stimuli, which is a supported use, not a mistake.
PARAM_SPECS = [
    ("age_days", "Chick age (days)", 2.0, 20.0, 0.5, (4, 15),
     "Age the call should sound like. Training data covers days 4-15."),
    ("sex", "Sex (0 = male, 1 = female)", -0.5, 1.5, 0.05, (0, 1),
     "0.5 is ambiguous. Values outside 0-1 push the decoder past either sex."),

    ("centroid_khz", "Brightness / centroid (kHz)", 1.0, 9.0, 0.1, (2.3, 5.5),
     "Where the call's energy sits. Higher sounds brighter and thinner."),
    ("f0_khz", "Pitch F0 (kHz)", 1.0, 9.0, 0.1, (1.87, 5.95),
     "Fundamental frequency. Moves independently of brightness, so pitch and "
     "brightness can be decoupled beyond what a real chick can do."),
    ("noisiness", "Noisiness", -0.5, 1.5, 0.05, (0, 1),
     "0 is tonal and whistle-like, 1 is harsh and noisy."),
    ("amplitude_dbfs", "Playback level (dBFS)", -40.0, 0.0, 1.0, None,
     "Level of the saved file. -22 is a moderate default; set the real loudness "
     "at the amplifier, not here."),

    ("duration_ms", "Call length (ms)", 20.0, 300.0, 1.0, (48.7, 162.9),
     "How long one call lasts."),
    ("calls_per_bout", "Calls per bout", 1, 40, 1, (1, 25),
     "How many calls in a row make up one bout."),
    ("ici_ms", "Gap between calls (ms)", 20.0, 500.0, 1.0, (52.9, 253.9),
     "Inter-call interval inside a bout."),
    ("isochrony", "Rhythm regularity", 0.0, 1.0, 0.01, (0.37, 0.95),
     "1 is a metronome; lower is more irregular timing."),
    ("n_bouts", "Number of bouts", 1, 10, 1, None,
     "How many bouts in the file."),
    ("ibi_s", "Gap between bouts (s)", 0.1, 5.0, 0.05, (0.31, 1.47),
     "Silence between bouts."),

    ("amp_jitter_db", "Loudness variation (dB)", 0.0, 12.0, 0.5, None,
     "How much loudness varies from call to call."),
    ("dur_jitter_frac", "Length variation", 0.0, 1.0, 0.01, (0.036, 0.653),
     "How much call length varies within a bout."),
    ("accent_prob", "Accented calls", 0.0, 1.0, 0.05, None,
     "Chance a call is emphasised: longer and louder."),
    ("accent_pos", "Where accents fall", 0.0, 1.0, 0.05, None,
     "0 is the start of the bout, 1 the end."),

    ("seed", "Random seed", 0, 9999, 1, None,
     "The same seed and settings always give the same call."),
    ("gl_iters", "Rendering quality", 10, 200, 5, None,
     "Griffin-Lim iterations. 50 is fine for listening; raise it for a final stimulus."),
]

GROUPS = [
    ("Which chick", ["age_days", "sex"]),
    ("How the call sounds", ["centroid_khz", "f0_khz", "noisiness", "amplitude_dbfs"]),
    ("Timing", ["duration_ms", "calls_per_bout", "ici_ms", "isochrony", "n_bouts", "ibi_s"]),
    ("Variation between calls", ["amp_jitter_db", "dur_jitter_frac", "accent_prob", "accent_pos"]),
    ("Reproducibility", ["seed", "gl_iters"]),
]

INT_PARAMS = {"calls_per_bout", "n_bouts", "seed", "gl_iters"}


def defaults():
    """The engine's own defaults, as a plain dict."""
    ge = engine()
    d = dict(ge.CallParams().__dict__)
    return {k: (int(v) if k in INT_PARAMS else v) for k, v in d.items()}


def typical_range(key):
    for spec in PARAM_SPECS:
        if spec[0] == key:
            return spec[5]
    return None


def is_supernormal(key, value):
    """True when a setting is outside what real chicks in the training set did."""
    rng = typical_range(key)
    if not rng:
        return False
    try:
        return not (rng[0] <= float(value) <= rng[1])
    except (TypeError, ValueError):
        return False


def supernormal_settings(params):
    return [k for k in params if is_supernormal(k, params[k])]


def render(params, with_audio=True):
    """Render one stimulus. Returns (waveform float32 at SR, list of spectrograms)."""
    ge = engine()
    p = dict(defaults())
    p.update({k: v for k, v in (params or {}).items() if k in p})
    for k in INT_PARAMS:
        if k in p:
            p[k] = int(round(float(p[k])))
    y, specs = ge.render(ge.CallParams(**p), with_audio=with_audio)
    return np.asarray(y, dtype=np.float32), specs


def filename_stem(params):
    ge = engine()
    p = dict(defaults()); p.update({k: v for k, v in params.items() if k in p})
    for k in INT_PARAMS:
        p[k] = int(round(float(p[k])))
    return ge.params_to_filename_stem(ge.CallParams(**p))


def save_stimulus(y, dest_dir, params, name=None, sr=SR, note=""):
    """Write a generated call into a stimulus folder, with its parameters recorded.

    Matches the folder's existing sample rate, exactly as a recorded clip does, so
    Design Experiment never sees a folder of mixed rates.
    """
    import song_library as slib
    os.makedirs(dest_dir, exist_ok=True)
    out_sr = int(slib.folder_sample_rate(dest_dir) or sr)
    audio = slib.resample_to(np.asarray(y, dtype=np.float64), sr, out_sr)
    audio = np.clip(audio, -1.0, 1.0)

    base = name or filename_stem(params)
    base = "".join(c for c in base if c.isalnum() or c in "._-") or "generated_call"
    dest = os.path.join(dest_dir, base + ".wav")
    i = 2
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{base}_{i}.wav")
        i += 1
    sf.write(dest, audio.astype(np.float32), out_sr, subtype="PCM_16")

    supern = supernormal_settings(params)
    slib._append_provenance(dest_dir, dict(
        stimulus=os.path.basename(dest),
        source="VAE generator (callgen)",
        start_s="", end_s="",
        duration_s=round(len(audio) / out_sr, 4),
        source_rate=sr, sample_rate=out_sr,
        normalize="generator", fade_ms="",
        note=(note + (f"  SUPERNORMAL: {', '.join(supern)}" if supern else "")).strip(),
        exported=time.strftime("%Y-%m-%d %H:%M:%S"),
        parameters=json.dumps({k: params[k] for k in sorted(params)}),
    ))
    return dest


def describe(y, sr=SR):
    """Measure a generated call with FEATHER's own analysis, as a sanity check.

    Worth doing: audio reconstructed from mel spectrograms by Griffin-Lim comes out
    measurably more tonal (lower Wiener entropy) than a real recording, so this is
    how you see what you are actually about to play.
    """
    import song_features as sfx
    feat, syl, _sp = sfx.analyze(np.asarray(y, dtype=np.float64), sr)
    if len(syl) == 0:
        return dict(calls=0)
    return dict(
        calls=int(len(syl)),
        median_duration_ms=float(syl["duration_ms"].median()),
        median_centre_hz=float(syl["mean_frequency_mean"].median()),
        median_entropy=float(syl["entropy_mean"].median()),
        median_fm=float(syl["fm_mean"].median()),
        seconds=float(len(y) / sr),
    )


def series(base_params, key, values, dest_dir, name_prefix=None, note="", progress=None):
    """Generate a graded set: one stimulus per value of `key`, everything else fixed.

    This is how a supernormal experiment is built — a ladder of stimuli that differ in
    exactly one property, sharing a seed so nothing else moves.
    """
    out = []
    for i, v in enumerate(values):
        if progress is not None:
            progress(i, len(values), v)
        p = dict(base_params); p[key] = v
        y, _ = render(p)
        stem = None
        if name_prefix:
            stem = f"{name_prefix}_{key}_{str(v).replace('.', 'p')}"
        out.append(save_stimulus(y, dest_dir, p, name=stem,
                                 note=(note or f"series over {key}")))
    if progress is not None:
        progress(len(values), len(values), None)
    return out
