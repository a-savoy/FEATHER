"""song_library.py — browse recordings, cache their measurements, cut stimuli.

The middle layer of Sound Bench: it finds recordings, runs song_features over them
once and remembers the answer, and turns a chosen stretch of audio into a stimulus
WAV that FEATHER's Design Experiment tab can pick up with no import step.

The cache is a single SQLite file (SAP used MySQL; a lab tool should not need a
database server). Re-analysing a folder you have already looked at is instant, and the
cache invalidates itself when a file changes or when the analysis parameters change.

Export is the one operation that writes into your stimulus folder, so it is careful:
it matches the sample rate already in use there, because playback_tools refuses a
folder whose WAVs disagree, and it records where every clip came from.
"""
from __future__ import annotations

import io
import os
import glob
import json
import time
import hashlib
import sqlite3

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

import song_features as sfx

CACHE_NAME = "sound_bench_cache.sqlite"
PROVENANCE = "stimulus_provenance.csv"
AUDIO_EXT = (".wav", ".flac", ".aiff", ".aif")


# ---------------------------------------------------------------------------
# Finding recordings
# ---------------------------------------------------------------------------
def scan_folder(folder, recursive=True, include_rejected=False):
    """Every audio file under `folder`, with its bird and day where the layout says so.

    Understands song_recorder's  <bird>/<YYYY-MM-DD>/<file>.wav  layout, and is happy
    with a flat folder of WAVs too.
    """
    folder = os.path.abspath(folder)
    pat = "**/*" if recursive else "*"
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, pat), recursive=recursive)):
        if not path.lower().endswith(AUDIO_EXT) or not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, folder)
        parts = rel.split(os.sep)
        if not include_rejected and "rejected" in parts:
            continue
        try:
            info = sf.info(path)
        except Exception:
            continue
        bird = parts[0] if len(parts) > 2 else ""
        day = parts[1] if len(parts) > 2 and len(parts[1]) == 10 and parts[1][4] == "-" else ""
        rows.append(dict(file=os.path.basename(path), path=path, bird=bird, day=day,
                         duration_s=round(info.duration, 3), sample_rate=info.samplerate,
                         channels=info.channels,
                         rejected=("rejected" in parts),
                         modified=os.path.getmtime(path)))
    return pd.DataFrame(rows, columns=["file", "path", "bird", "day", "duration_s",
                                       "sample_rate", "channels", "rejected", "modified"])


def read_audio(path, mono=True):
    x, fs = sf.read(path, always_2d=False)
    if mono and getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float64), float(fs)


# ---------------------------------------------------------------------------
# Cached analysis
# ---------------------------------------------------------------------------
def _params_key(spec_params, segment_params):
    p = dict(spec=dict(sfx.SPEC_DEFAULTS, **(spec_params or {})),
             seg=dict(sfx.SEGMENT_DEFAULTS, **(segment_params or {})))
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Library:
    """Syllable tables for a set of recordings, computed once and cached."""

    def __init__(self, cache_path):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS analysis (
                             path TEXT, params TEXT, mtime REAL, size INTEGER,
                             analysed_at REAL, syllables TEXT,
                             PRIMARY KEY (path, params))""")
        self.db.commit()

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    def analyze(self, path, spec_params=None, segment_params=None, force=False):
        """Syllable table for one file, from cache when the file and settings match."""
        key = _params_key(spec_params, segment_params)
        st = os.stat(path)
        if not force:
            row = self.db.execute(
                "SELECT mtime, size, syllables FROM analysis WHERE path=? AND params=?",
                (path, key)).fetchone()
            if row and abs(row[0] - st.st_mtime) < 1e-6 and row[1] == st.st_size:
                # pandas 3 treats a bare JSON string as a path; wrap it.
                return pd.read_json(io.StringIO(row[2]), orient="split")

        x, fs = read_audio(path)
        _feat, syl, _sp = sfx.analyze(x, fs, spec_params=spec_params,
                                      segment_params=segment_params)
        self.db.execute("REPLACE INTO analysis VALUES (?,?,?,?,?,?)",
                        (path, key, st.st_mtime, st.st_size, time.time(),
                         syl.to_json(orient="split")))
        self.db.commit()
        return syl

    def analyze_many(self, paths, spec_params=None, segment_params=None,
                     progress=None, stop=None):
        """Analyze a list of files. progress(i, n, path) is called as it goes;
        stop() returning True aborts. Returns one table with a `file` column."""
        out = []
        n = len(paths)
        for i, path in enumerate(paths):
            if stop is not None and stop():
                break
            if progress is not None:
                progress(i, n, path)
            try:
                syl = self.analyze(path, spec_params, segment_params)
            except Exception as e:
                out.append(pd.DataFrame([dict(file=os.path.basename(path), path=path,
                                              error=f"{type(e).__name__}: {e}")]))
                continue
            if len(syl):
                syl = syl.copy()
                syl.insert(0, "path", path)
                syl.insert(0, "file", os.path.basename(path))
                out.append(syl)
        if progress is not None:
            progress(n, n, "")
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def summary(self, table):
        """One row per file: how many syllables, and their median feature values."""
        if table is None or table.empty or "syllable" not in table.columns:
            return pd.DataFrame()
        g = table.groupby("file")
        out = pd.DataFrame({"file": g.size().index, "syllables": g.size().to_numpy()})
        out["median_duration_ms"] = g["duration_ms"].median().to_numpy()
        for f in ("mean_frequency", "entropy", "goodness", "fm"):
            col = f"{f}_mean"
            if col in table.columns:
                out[f] = g[col].median().to_numpy().round(3)
        return out


# ---------------------------------------------------------------------------
# Cutting stimuli
# ---------------------------------------------------------------------------
def folder_sample_rate(dest_dir):
    """The rate already in use in a stimulus folder, or None if it is empty.

    playback_tools rejects a folder whose WAVs disagree, so a new clip must match.
    """
    rates = set()
    for f in glob.glob(os.path.join(dest_dir, "*.wav")):
        try:
            rates.add(sf.info(f).samplerate)
        except Exception:
            pass
    if not rates:
        return None
    if len(rates) > 1:
        raise ValueError(f"{dest_dir} already holds mixed sample rates: {sorted(rates)}. "
                         "Re-export those files at one rate before adding more.")
    return int(rates.pop())


def resample_to(x, fs, target_fs):
    if int(fs) == int(target_fs):
        return x
    from math import gcd
    g = gcd(int(fs), int(target_fs))
    return resample_poly(x, int(target_fs) // g, int(fs) // g)


def export_stimulus(src_path, dest_dir, start_s=None, end_s=None, name=None,
                    fade_ms=5.0, normalize="peak", peak_level=0.89,
                    target_fs=None, note=""):
    """Cut a clip out of a recording and write it into a stimulus folder.

    Applies a short fade at both ends (a hard cut clicks, and a click is a stimulus
    onset the bird can hear), normalises, and matches the folder's sample rate.
    Returns the path written. Appends a provenance row so every stimulus can be
    traced back to the recording and offsets it came from.
    """
    os.makedirs(dest_dir, exist_ok=True)
    x, fs = read_audio(src_path)
    n = len(x)
    a = 0 if start_s is None else max(0, int(round(start_s * fs)))
    b = n if end_s is None else min(n, int(round(end_s * fs)))
    if b - a < 8:
        raise ValueError("Selection is too short to export.")
    clip = np.array(x[a:b], dtype=np.float64)

    if fade_ms and fade_ms > 0:
        r = min(int(fade_ms * 1e-3 * fs), len(clip) // 2)
        if r > 1:
            clip[:r] *= np.linspace(0.0, 1.0, r)
            clip[-r:] *= np.linspace(1.0, 0.0, r)

    if normalize == "peak":
        m = np.max(np.abs(clip))
        if m > 0:
            clip *= peak_level / m
    elif normalize == "rms":
        r = np.sqrt(np.mean(clip ** 2))
        if r > 0:
            clip *= 0.1 / r
            clip = np.clip(clip, -0.99, 0.99)

    out_fs = int(target_fs or folder_sample_rate(dest_dir) or fs)
    clip = resample_to(clip, fs, out_fs)
    clip = np.clip(clip, -1.0, 1.0)

    base = name or os.path.splitext(os.path.basename(src_path))[0]
    if start_s is not None:
        # Name from the clip's ACTUAL start, which may have been clamped to 0.
        base = f"{base}_{int(round(a / fs * 1000))}ms"
    base = "".join(c for c in base if c.isalnum() or c in "._-") or "stimulus"
    dest = os.path.join(dest_dir, base + ".wav")
    i = 2
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{base}_{i}.wav")
        i += 1
    sf.write(dest, clip.astype(np.float32), out_fs, subtype="PCM_16")

    _append_provenance(dest_dir, dict(
        stimulus=os.path.basename(dest), source=os.path.abspath(src_path),
        start_s=round(a / fs, 4), end_s=round(b / fs, 4),
        duration_s=round(len(clip) / out_fs, 4), source_rate=int(fs), sample_rate=out_fs,
        normalize=normalize, fade_ms=fade_ms, note=note,
        exported=time.strftime("%Y-%m-%d %H:%M:%S")))
    return dest


def _append_provenance(dest_dir, row):
    path = os.path.join(dest_dir, PROVENANCE)
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def provenance(dest_dir):
    path = os.path.join(dest_dir, PROVENANCE)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
