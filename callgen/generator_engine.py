"""
Generator engine for the zebra finch begging-call synthesizer.

Conditions the VAE on (age, duration, sex) and shifts the latent along
discovered direction vectors (centroid, F0, noisiness) to give intuitive
slider control. Supernormal values (outside the training distribution)
are passed through unclipped for research use.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from train_vae import DAY_MIN, DAY_MAX, DUR_MIN, DUR_MAX, TIME_PAD_TO, FLOOR, CVAE
from audio_utils import (
    logmel_to_waveform, loudness_normalize, apply_edge_fade,
    impose_duration, trim_to_active, SR, HOP,
)

HERE = Path(__file__).resolve().parent
CKPT = HERE / "vae_ckpt.pt"
NORM = HERE / "vae_norm_stats.npz"
DIRS = HERE / "latent_directions.npz"


@dataclass
class CallParams:
    # Ethological
    age_days: float = 10.0            # middle of training range (4–15)
    sex: float = 0.5                 # 0 = M, 1 = F (ambiguous by default)
    # Spectral — middle of empirical species-typical ranges
    amplitude_dbfs: float = -22.0    # moderate playback level
    centroid_khz: float = 3.9        # middle of 2.3–5.5 kHz
    f0_khz: float = 3.9              # middle of 1.9–6.0 kHz
    noisiness: float = 0.5           # middle of species-typical 0–1
    # Single-call temporal — middle of 49–163 ms
    duration_ms: float = 105.0
    # Bout-level temporal — middles of species-typical ranges
    calls_per_bout: int = 13         # middle of 1–25
    ici_ms: float = 154.0            # middle of 53–254 ms
    isochrony: float = 0.65          # middle of 0.37–0.95
    n_bouts: int = 2                 # moderate
    ibi_s: float = 0.9               # middle of 0.31–1.47 s
    # Within-bout prosody — moderate / middles
    amp_jitter_db: float = 3.0        # moderate amp variation across calls in a bout
    dur_jitter_frac: float = 0.34    # middle of 0.04–0.65
    accent_prob: float = 0.2          # moderate — some accented calls, not most
    accent_pos: float = 0.5           # centered
    # Generation controls
    seed: int = 0
    gl_iters: int = 50                 # live-preview quality; faster on HF CPU


_STATE = {"model": None, "mean": None, "std": None, "dirs": None}


def _load():
    if _STATE["model"] is not None:
        return
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = CVAE(latent=d["latent"], cond_dim=d["cond_dim"])
    model.load_state_dict(d["model"])
    model.eval()
    norm = np.load(NORM)
    dirs = np.load(DIRS)
    _STATE["model"] = model
    _STATE["mean"] = float(norm["mean"])
    _STATE["std"] = float(norm["std"])
    _STATE["dirs"] = {k: np.asarray(v) for k, v in dict(dirs).items()}


def get_state():
    _load()
    return _STATE


# ------------------------ parameter → model ------------------------

def build_cond(params: CallParams) -> torch.Tensor:
    """cond = [age_n, dur_n, sex]. Sex is passed through unclipped so
    supernormal values outside [0, 1] drive the decoder into extrapolation."""
    age_n = float(np.clip((params.age_days - DAY_MIN) / (DAY_MAX - DAY_MIN), 0.0, 1.0))
    dur_n = float(np.clip(
        (params.duration_ms / 1000.0 - DUR_MIN) / (DUR_MAX - DUR_MIN), 0.0, 1.0))
    sex_n = float(params.sex)   # unclipped for supernormal exploration
    vec = np.array([age_n, dur_n, sex_n], dtype=np.float32)
    return torch.from_numpy(vec[None, :])


def build_z(params: CallParams, call_idx: int) -> torch.Tensor:
    """Start from a perturbed empirical mean, then shift along
    centroid_dir, f0_dir_orth, and noisiness_dir by amounts derived
    from the user's slider values.

    F0 uses the ORTHOGONALIZED direction (f0_dir_orth) so that moving
    the F0 slider doesn't also drag the centroid — enabling supernormal
    decoupling of naturally-correlated properties."""
    _load()
    dirs = _STATE["dirs"]
    emp_mean = dirs["empirical_z_mean"]
    cent_dir = dirs["centroid_dir"]
    cent_scale = float(dirs["centroid_dir_scale"])
    f0_dir_orth = dirs["f0_dir_orth"]
    f0_scale = float(dirs["f0_dir_scale"])
    noise_dir = dirs["noisiness_dir"]
    noise_scale = float(dirs["noisiness_dir_scale"])
    cent_rng = dirs["centroid_range"]
    f0_rng = dirs["f0_range"]
    noise_rng = dirs["noisiness_range"]

    rng = np.random.default_rng(params.seed + call_idx * 9973)
    base = emp_mean + 0.40 * rng.standard_normal(emp_mean.shape).astype(np.float32)

    # Centroid: slider kHz → Hz offset from midpoint → z-units along centroid_dir
    mean_cent = float(0.5 * (cent_rng[0] + cent_rng[1]))
    alpha_cent = (params.centroid_khz * 1000.0 - mean_cent) / (cent_scale + 1e-12)

    # F0: along the orthogonalized direction. Note f0_scale is the norm of
    # the RAW f0 regression coef (Hz per raw-f0_dir unit). Since f0_dir_orth
    # is unit-norm but only a projection of the raw f0_dir, moving 1 unit
    # along f0_dir_orth changes predicted F0 by roughly f0_scale * retention.
    mean_f0 = float(0.5 * (f0_rng[0] + f0_rng[1]))
    retention = float(dirs.get("f0_dir_orth_retention", 0.667))
    f0_effective_scale = f0_scale * max(retention, 0.1)
    alpha_f0 = (params.f0_khz * 1000.0 - mean_f0) / (f0_effective_scale + 1e-12)

    # Noisiness: slider 0–1 maps to 5–95% log-entropy range
    target_ent = noise_rng[0] + params.noisiness * (noise_rng[1] - noise_rng[0])
    mean_ent = float(0.5 * (noise_rng[0] + noise_rng[1]))
    beta_noise = (target_ent - mean_ent) / (noise_scale + 1e-12)

    z = (base
         + alpha_cent * cent_dir
         + alpha_f0 * f0_dir_orth
         + beta_noise * noise_dir)
    return torch.from_numpy(z[None, :].astype(np.float32))


def _decode_one(params: CallParams, call_idx: int, with_audio: bool,
                 duration_ms_override: float | None = None):
    _load()
    model = _STATE["model"]
    mean, std = _STATE["mean"], _STATE["std"]
    if duration_ms_override is not None:
        cond_params = CallParams(**{**params.__dict__, "duration_ms": duration_ms_override})
    else:
        cond_params = params
    cond = build_cond(cond_params)
    z = build_z(params, call_idx)
    with torch.no_grad():
        x_hat = model.decoder(z, cond)
    spec = x_hat.cpu().numpy().squeeze() * std + mean
    spec = spec[:, :61]
    use_dur = (duration_ms_override if duration_ms_override is not None
               else params.duration_ms)
    spec = impose_duration(spec, use_dur / 1000.0, floor=FLOOR)
    spec_trim = trim_to_active(spec, floor=FLOOR, pad_frames=4)
    if not with_audio:
        return spec_trim, None
    y = logmel_to_waveform(spec_trim, n_iter=params.gl_iters)
    return spec_trim, y


def _sequence_bout(calls, ici_ms, isochrony, seed):
    if len(calls) == 1:
        bout = calls[0]
    else:
        rng = np.random.default_rng(seed * 7919 + 101)
        mean_gap = ici_ms / 1000.0
        jitter_sigma = (1.0 - float(np.clip(isochrony, 0.0, 1.0))) * 0.30 * mean_gap
        pieces = [calls[0]]
        for c in calls[1:]:
            gap = max(0.005, mean_gap + jitter_sigma * rng.standard_normal())
            pieces.append(np.zeros(int(round(gap * SR)), dtype=np.float32))
            pieces.append(c)
        bout = np.concatenate(pieces)
    # No bout-level fade. Earlier 40 ms / 100 ms attempts forced the last
    # call's amplitude through a cosine envelope down to zero — heard as
    # the call being truncated mid-utterance, which was the actual source
    # of the perceived "click" at the bout-to-IBI boundary. The per-call
    # 8 ms fade in logmel_to_waveform already lands every call at exactly
    # amp=0, so the bout's final sample is zero without further work.
    return bout


def _sequence_bouts(bouts, ibi_s):
    if len(bouts) == 1:
        return bouts[0]
    gap = np.zeros(int(round(ibi_s * SR)), dtype=np.float32)
    pieces = [bouts[0]]
    for b in bouts[1:]:
        pieces.append(gap)
        pieces.append(b)
    return np.concatenate(pieces)


def _bout_prosody(params: CallParams, bout_idx: int):
    N = int(params.calls_per_bout)
    rng = np.random.default_rng(params.seed + bout_idx * 1009)
    durs = np.full(N, float(params.duration_ms), dtype=np.float32)
    amps_db = np.zeros(N, dtype=np.float32)
    if N <= 1:
        return durs, amps_db
    if params.dur_jitter_frac > 0:
        # Truncate to ±1.5σ before applying as jitter. An unbounded normal
        # produces ~one -2σ outlier per 40 calls, which on default settings
        # (105 ms × 0.34 jitter) collapses a single call to ~20 ms — the
        # VAE renders that as a click-shaped impulse, perceived as the
        # bout's last call being "cut off". ±1.5σ aligns the truncated
        # distribution with the empirical 5th–95th percentile range.
        jit = np.clip(rng.standard_normal(N), -1.5, 1.5).astype(np.float32)
        # Final call uses the unjittered mean duration. A jittered-short
        # final call sounds truncated relative to its neighbors — listener
        # hears the motif ending mid-utterance rather than completing.
        jit[-1] = 0.0
        durs = durs * (1.0 + params.dur_jitter_frac * jit)
    if params.amp_jitter_db > 0:
        amps_db = amps_db + params.amp_jitter_db * rng.standard_normal(N).astype(np.float32)
    if params.accent_prob > 0:
        positions = np.linspace(0.0, 1.0, N).astype(np.float32)
        weights = np.exp(-((positions - float(params.accent_pos)) / 0.18) ** 2)
        mean_w = float(weights.mean()) + 1e-9
        probs = np.clip(weights * params.accent_prob / mean_w, 0.0, 1.0)
        is_accent = rng.random(N).astype(np.float32) < probs
        durs = durs * np.where(is_accent, 1.40, 1.0)
        amps_db = amps_db + np.where(is_accent, 6.0, 0.0)
    durs = np.clip(durs, 20.0, 250.0)
    return durs.astype(np.float32), amps_db.astype(np.float32)


def render(params: CallParams, with_audio: bool = True):
    _load()
    cpb = int(params.calls_per_bout)
    specs = []
    bout_waves = []

    for b in range(int(params.n_bouts)):
        durs, amps_db = _bout_prosody(params, b)
        bout_calls = []
        for i in range(cpb):
            call_idx = b * cpb + i
            spec, y = _decode_one(params, call_idx, with_audio=with_audio,
                                    duration_ms_override=float(durs[i]))
            specs.append(spec)
            if with_audio:
                gain = 10.0 ** (float(amps_db[i]) / 20.0)
                bout_calls.append((y * gain).astype(np.float32))
        if with_audio:
            bout_waves.append(_sequence_bout(bout_calls, params.ici_ms,
                                              params.isochrony, params.seed + b))

    if not with_audio:
        return np.empty(0, dtype=np.float32), specs

    full = _sequence_bouts(bout_waves, params.ibi_s)
    peak = float(np.max(np.abs(full)))
    if peak > 0:
        full = full / peak
    gain = 10 ** (float(params.amplitude_dbfs) / 20.0)
    full = full * gain
    # Literal-silence head/tail so no player can ever hit content at t=0.
    pad = np.zeros(int(0.020 * SR), dtype=np.float32)
    full = np.concatenate([pad, full, pad])
    return full.astype(np.float32), specs


def params_to_filename_stem(p: CallParams) -> str:
    return (f"zfcall_age{p.age_days:.0f}_sex{p.sex:.2f}"
            f"_dur{p.duration_ms:.0f}_cent{p.centroid_khz:.1f}k"
            f"_f0{p.f0_khz:.1f}k"
            f"_n{p.calls_per_bout}x{p.n_bouts}_ICI{p.ici_ms:.0f}"
            f"_iso{p.isochrony:.1f}_noise{p.noisiness:.1f}"
            f"_amp{p.amplitude_dbfs:.0f}dBFS_seed{p.seed}")


if __name__ == "__main__":
    import soundfile as sf
    import time
    p = CallParams()
    t0 = time.time()
    y, specs = render(p, with_audio=True)
    dt = time.time() - t0
    print(f"default params: {len(y)/SR*1000:.0f} ms audio, {len(specs)} call(s), "
          f"rendered in {dt*1000:.0f} ms")
    out = HERE / "audio" / f"{params_to_filename_stem(p)}.wav"
    out.parent.mkdir(exist_ok=True)
    sf.write(str(out), y, SR)
    print(f"saved {out}")
