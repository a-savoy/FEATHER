"""feather_analysis.py — the analysis half of FEATHER as plain, importable functions.

These mirror the tested logic inside playback_gui.py so the notebook (or your own
scripts) can run the identical pipeline without the GUI:

    load an Open Ephys recording -> detect stimulus onsets from the ADC1 gate
    -> identify each sound and its time offset from the ADC2 audio copy
    -> check timing -> spike-sort with Kilosort4 -> PSTH.

See HARDWARE.md for the ADC channel wiring these functions assume.
"""
import os
import glob
import json
import numpy as np

# ---- channel wiring (see HARDWARE.md) ----
GATE_CHANNEL = "ADC1"    # digital "sound-playing" gate; rising edge = stimulus onset
AUDIO_CHANNEL = "ADC2"   # copy of the played audio (identity + exact time)
MIC_CHANNEL = "ADC3"     # optional microphone at the animal

# ---- Kilosort4 defaults (same as the app's fixed defaults) ----
KS_PARAM_DEFAULTS = dict(
    Th_universal=9.0, Th_learned=8.0, Th_single_ch=6.0, nblocks=1,
    batch_size=60000, highpass_cutoff=300, dminx=32, min_template_size=10,
    nearest_chans=10, duplicate_spike_ms=0.25,
)


def find_recording(root):
    """Newest Open Ephys 'recording' folder under `root` (the dir holding a
    structure.oebin). Returns a path or None."""
    hits = glob.glob(os.path.join(root, "**", "structure.oebin"), recursive=True)
    if not hits:
        return None
    return os.path.dirname(max(hits, key=os.path.getmtime))


def load_adc_stream(recdir):
    """Load the continuous stream that carries the ADC sync channels.
    Returns dict(data=memmap[n,nch], names=[...], fs=float, n=int)."""
    oeb = sorted(glob.glob(os.path.join(recdir, "**", "structure.oebin"), recursive=True))[0]
    meta = json.load(open(oeb))
    base = os.path.dirname(oeb)
    streams = meta["continuous"]
    s = next((x for x in streams
              if any(c["channel_name"] == GATE_CHANNEL for c in x["channels"])), streams[0])
    names = [c["channel_name"] for c in s["channels"]]
    nch = int(s["num_channels"])
    fs = float(s["sample_rate"])
    dat = os.path.join(base, "continuous", s["folder_name"], "continuous.dat")
    if not os.path.exists(dat):
        dat = glob.glob(os.path.join(base, "**", "continuous.dat"), recursive=True)[0]
    n = os.path.getsize(dat) // (2 * nch)
    data = np.memmap(dat, dtype="<i2", mode="r", shape=(n, nch))
    return dict(data=data, names=names, fs=fs, n=n)


def detect_onsets(data, names, fs):
    """Rising/falling edges of the ADC1 gate. Returns (onsets, offsets) as sample
    indices (rising edge = sample-accurate stimulus onset)."""
    gate = data[:, names.index(GATE_CHANNEL)].astype(np.float64)
    mid = (np.percentile(gate, 1) + np.percentile(gate, 99)) / 2
    above = gate > mid

    def dedup(idx):
        out, last = [], -10 ** 9
        for r in idx:
            if r - last > int(0.2 * fs):
                out.append(int(r)); last = r
        return out

    onsets = dedup(np.where((~above[:-1]) & (above[1:]))[0] + 1)
    offsets = dedup(np.where((above[:-1]) & (~above[1:]))[0] + 1)
    return onsets, offsets


def identify_by_audio(data, names, fs, onsets, stimuli_dir):
    """For each onset, identify which stimulus WAV played and the TIME OFFSET (ms)
    between the timing marker and the recorded audio, via a lag-aligned normalized
    cross-correlation (+-60 ms search, parabolic sub-sample refinement).

    Returns a list of (best_file, offset_ms, corr) per onset. offset_ms > 0 means the
    recorded audio arrives AFTER the marker; corr (0..1) < ~0.3 means no sound matched.
    """
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly, correlate
    fsi = int(round(fs))
    audio = data[:, names.index(AUDIO_CHANNEL)].astype(np.float64)
    stims = {}
    for f in sorted(os.listdir(stimuli_dir)):
        if f.lower().endswith(".wav"):
            x, sr = sf.read(os.path.join(stimuli_dir, f), dtype="float64", always_2d=True)
            x = x[:, 0]; sr = int(sr)
            if sr != fsi:
                g = gcd(fsi, sr); x = resample_poly(x, fsi // g, sr // g)
            stims[f] = x - x.mean()
    seglen = int(5.2 * fs)
    maxlag = int(0.060 * fs)
    out = []
    for o in onsets:
        seg = audio[o:o + seglen].astype(np.float64)
        seg = seg - seg.mean()
        best_f, best_c, best_lag = None, -1e9, 0.0
        for f, t in stims.items():
            L = min(len(seg), len(t))
            a = seg[:L] / (np.linalg.norm(seg[:L]) + 1e-9)
            b = t[:L] / (np.linalg.norm(t[:L]) + 1e-9)
            c = correlate(a, b, mode="full")
            lags = np.arange(-(L - 1), L)
            cc = np.where(np.abs(lags) <= maxlag, c, -np.inf)
            i = int(np.argmax(cc))
            if c[i] > best_c:
                delta = 0.0
                if 0 < i < len(c) - 1:
                    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
                    den = y0 - 2 * y1 + y2
                    if den != 0:
                        delta = 0.5 * (y0 - y2) / den
                best_c = float(c[i]); best_f = f
                best_lag = (lags[i] + delta) / fs * 1000.0
        out.append((best_f, best_lag, best_c))
    return out


def timing_check(onsets, offsets, fs, audio_id, stimuli_dir):
    """Residual (ms) between each gate marker width and the true stimulus duration.
    Small mean + sub-ms SD == sample-accurate sync. Returns a numpy array."""
    import soundfile as sf
    resid, durcache = [], {}
    for k, o in enumerate(onsets):
        off = next((x for x in offsets if x > o), None)
        if off is None:
            continue
        w = (off - o) / fs
        f = audio_id[k][0] if audio_id else None
        if f and f not in durcache:
            try:
                durcache[f] = sf.info(os.path.join(stimuli_dir, f)).duration
            except Exception:
                durcache[f] = None
        if f and durcache[f]:
            resid.append((w - durcache[f]) * 1000.0)
    return np.array(resid)


def find_neural_stream(recdir):
    """The continuous stream with >=16 non-ADC/MEM channels (the probe data).
    Returns dict(dat, n_chan, fs) or None."""
    oeb = None
    for dp, _, fn in os.walk(recdir or ""):
        if "structure.oebin" in fn:
            oeb = os.path.join(dp, "structure.oebin")
    if oeb is None:
        return None
    meta = json.load(open(oeb))
    base = os.path.dirname(oeb)
    for s in meta["continuous"]:
        nm = [c["channel_name"] for c in s["channels"]]
        neural = [i for i, x in enumerate(nm) if not (x.startswith("ADC") or x == "MEM")]
        if len(neural) >= 16:
            dat = os.path.join(base, "continuous", s["folder_name"], "continuous.dat")
            if not os.path.exists(dat):
                g = glob.glob(os.path.join(base, "**", "continuous.dat"), recursive=True)
                dat = g[0] if g else None
            return dict(dat=dat, n_chan=int(s["num_channels"]), fs=float(s["sample_rate"]))
    return None


def run_sorting(recdir, probe_file, params=None, device=None):
    """Run Kilosort4 on the recording's neural stream using a probeinterface probe
    file. Writes to <recdir>/kilosort4 and returns that path."""
    import torch
    from probeinterface import read_probeinterface
    from kilosort import run_kilosort
    ns = find_neural_stream(recdir)
    if ns is None:
        raise RuntimeError("No neural/probe channels in this recording (ADC-only?).")
    p = dict(KS_PARAM_DEFAULTS)
    if params:
        p.update(params)
    pr = read_probeinterface(probe_file).probes[0]
    pos = pr.contact_positions
    dev = pr.device_channel_indices
    sh = pr.shank_ids
    probe_ks = dict(
        chanMap=np.asarray(dev, dtype="int32"),
        xc=pos[:, 0].astype("float32"), yc=pos[:, 1].astype("float32"),
        kcoords=(np.asarray(sh, dtype=int).astype("float32")
                 if sh is not None else np.zeros(len(pos), "float32")),
        n_chan=int(ns["n_chan"]),
    )
    ks = os.path.join(recdir, "kilosort4")
    run_kilosort(settings=dict(n_chan_bin=int(ns["n_chan"]), fs=ns["fs"], **p),
                 probe=probe_ks, filename=ns["dat"], results_dir=ks,
                 data_dtype="int16",
                 device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    return ks


def load_kilosort(ks_dir):
    """Load a Kilosort4/Phy output folder.
    Returns dict(spike_times, spike_clusters, fs, units)."""
    spk = np.load(os.path.join(ks_dir, "spike_times.npy")).ravel().astype(np.int64)
    clu = np.load(os.path.join(ks_dir, "spike_clusters.npy")).ravel()
    _p = {}
    exec(open(os.path.join(ks_dir, "params.py")).read(), _p)
    fs = float(_p.get("sample_rate", 30000))
    return dict(spike_times=spk, spike_clusters=clu, fs=fs, units=np.unique(clu))


def responsiveness(ks, onsets, unit, base=(-0.25, -0.05), resp=(0.05, 0.25)):
    """Firing-rate change (Hz) in a post-onset window vs a pre-onset baseline."""
    st = ks["spike_times"][ks["spike_clusters"] == unit]
    fs = ks["fs"]

    def rate(a, b):
        return sum(np.sum((st >= o + int(a * fs)) & (st < o + int(b * fs))) for o in onsets) / len(onsets) / (b - a)

    return rate(*resp) - rate(*base)


def psth(ks, unit, onsets, pre=0.2, post=0.5):
    """Raster (top) + histogram (bottom) of one unit around each sound onset.
    Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    st = ks["spike_times"][ks["spike_clusters"] == unit]
    fs = ks["fs"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    allrel = []
    for row, o in enumerate(onsets):
        rel = (st[(st >= o - pre * fs) & (st < o + post * fs)] - o) / fs
        allrel.append(rel)
        ax1.plot(rel, np.full(rel.shape, row, float), "|", color="k", ms=6)
    ax1.axvline(0, color="r")
    ax1.set(ylabel="sound (trial)", title=f"Unit {unit}: response to sound onset")
    flat = np.concatenate(allrel) if allrel else np.array([])
    ax2.hist(flat, bins=np.arange(-pre, post, 0.02), color="#334155")
    ax2.axvline(0, color="r")
    ax2.set(xlabel="time from sound onset (s)", ylabel="spikes")
    fig.tight_layout()
    return fig
