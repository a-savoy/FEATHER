"""
Convert log-mel spectrograms back to listenable waveforms.

Path: log-mel -> mel-power -> linear-power (pinv of filterbank, clipped nonneg)
      -> magnitude -> Griffin-Lim phase recovery -> ISTFT -> waveform

Matches the analysis params used by build_melspec_dataset.py exactly.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import stft, istft

# Must match build_melspec_dataset.py
SR = 44100
N_FFT = 1024
HOP = 128
N_MELS = 64
FMIN, FMAX = 250.0, 10000.0


def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if ctr == lo: ctr = lo + 1
        if hi == ctr: hi = ctr + 1
        for k in range(lo, ctr):
            fb[m - 1, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi):
            fb[m - 1, k] = (hi - k) / max(hi - ctr, 1)
        width_hz = hz_pts[m + 1] - hz_pts[m - 1]
        if width_hz > 0:
            fb[m - 1] *= 2.0 / width_hz
    return fb


def mel_pseudo_inverse(fb: np.ndarray) -> np.ndarray:
    """Pseudo-inverse so that fb_inv @ (fb @ x) ≈ x for x >= 0."""
    return np.linalg.pinv(fb).astype(np.float32)


def logmel_to_linpower(log_mel: np.ndarray, fb_inv: np.ndarray) -> np.ndarray:
    """log_mel: (n_mels, T). Returns linear power spec (n_fft/2+1, T), nonneg."""
    mel_power = np.exp(log_mel) - 1e-10
    mel_power = np.maximum(mel_power, 0.0).astype(np.float32)
    lin = fb_inv @ mel_power
    return np.maximum(lin, 0.0)


def griffin_lim(mag: np.ndarray, n_fft=N_FFT, hop=HOP, n_iter=64, seed=0):
    """
    Phase reconstruction from magnitude spectrogram using Griffin-Lim.
    mag: (n_fft/2+1, T) magnitude spectrogram.
    Returns waveform np.ndarray.
    """
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(mag.shape).astype(np.float32))
    S = mag.astype(np.complex64) * phase
    for _ in range(n_iter):
        _, y = istft(S, fs=SR, nperseg=n_fft, noverlap=n_fft - hop,
                     window="hann", input_onesided=True, boundary=None)
        _, _, Sn = stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop,
                        window="hann", return_onesided=True,
                        padded=False, boundary=None)
        # Trim/pad to match mag shape
        if Sn.shape[1] < mag.shape[1]:
            pad = mag.shape[1] - Sn.shape[1]
            Sn = np.pad(Sn, ((0, 0), (0, pad)))
        else:
            Sn = Sn[:, : mag.shape[1]]
        new_phase = np.exp(1j * np.angle(Sn))
        S = mag.astype(np.complex64) * new_phase.astype(np.complex64)
    _, y = istft(S, fs=SR, nperseg=n_fft, noverlap=n_fft - hop,
                 window="hann", input_onesided=True, boundary=None)
    return y.astype(np.float32)


def apply_edge_fade(y: np.ndarray, sr: int = SR, fade_ms: float = 5.0) -> np.ndarray:
    """Raised-cosine fade-in/out to zero out ISTFT edge artifacts.

    For signals shorter than 2*fade_ms the fade length is clamped to half the
    buffer rather than skipped — a click-free taper matters more than hitting
    the requested duration exactly."""
    n_fade = int(sr * fade_ms / 1000.0)
    n_fade = min(n_fade, len(y) // 2)
    if n_fade < 2:
        return y
    fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade, dtype=np.float32) / n_fade))
    out = y.copy()
    out[:n_fade] *= fade
    out[-n_fade:] *= fade[::-1]
    return out


def trim_to_active(log_mel: np.ndarray, floor: float = -23.0259,
                    pad_frames: int = 2) -> np.ndarray:
    """Crop the spectrogram to its contiguous non-floor region (the call),
    with a small symmetric padding on either side. Used to strip leading/
    trailing silence before Griffin-Lim inversion."""
    # A frame is "active" if ANY mel bin is above the floor.
    active = (log_mel > floor + 1e-3).any(axis=0)
    if not active.any():
        return log_mel
    idx = np.where(active)[0]
    lo = max(0, idx[0] - pad_frames)
    hi = min(log_mel.shape[1], idx[-1] + 1 + pad_frames)
    return log_mel[:, lo:hi]


def impose_duration(log_mel: np.ndarray, duration_s: float,
                     sr: int = SR, hop: int = HOP, floor: float = -23.0259,
                     fade_ms: float = 10.0) -> np.ndarray:
    """Crop the spectrogram to `duration_s` around its peak-energy frame.
    Outside-call frames fade to floor with a raised-cosine."""
    e = log_mel.mean(axis=0)  # per-frame mean log-mel energy
    peak = int(np.argmax(e))
    frames_per_s = sr / hop
    half = int(round(duration_s * frames_per_s / 2))
    lo = max(0, peak - half)
    hi = min(log_mel.shape[1], peak + half + 1)
    fade_f = max(1, int(round(fade_ms / 1000.0 * frames_per_s)))
    mask = np.zeros(log_mel.shape[1], dtype=np.float32)
    mask[lo:hi] = 1.0
    if fade_f * 2 < (hi - lo):
        fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_f, dtype=np.float32) / fade_f))
        mask[lo:lo + fade_f] *= fade
        mask[hi - fade_f:hi] *= fade[::-1]
    return floor + mask[None, :] * (log_mel - floor)


def logmel_to_waveform(log_mel: np.ndarray, n_iter: int = 150,
                       fade_ms: float = 8.0) -> np.ndarray:
    fb = build_mel_filterbank()
    fb_inv = mel_pseudo_inverse(fb)
    lin_power = logmel_to_linpower(log_mel, fb_inv)
    mag = np.sqrt(lin_power + 1e-12)
    y = griffin_lim(mag, n_iter=n_iter)
    y = apply_edge_fade(y, SR, fade_ms=fade_ms)
    return y


def loudness_normalize(y: np.ndarray, target_rms_dbfs: float = -26.0,
                        peak_ceiling: float = 0.6) -> np.ndarray:
    """RMS-normalize to a target loudness (dBFS) with a peak safety cap.
    -26 dBFS RMS is a comfortable listening level for short noisy clips.
    """
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms <= 0:
        return y
    target = 10 ** (target_rms_dbfs / 20.0)
    out = (target / rms) * y
    peak = float(np.max(np.abs(out)))
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(np.float32)


if __name__ == "__main__":
    # Sanity check: round-trip a random spectrogram to audio
    import soundfile as sf
    from pathlib import Path
    here = Path(__file__).resolve().parent
    d = np.load(here / "melspec_dataset.npz")
    ms = d["melspecs"][0]  # first call
    print(f"mel shape {ms.shape}, range {ms.min():.2f}..{ms.max():.2f}")
    y = logmel_to_waveform(ms, n_iter=80)
    out = here / "plots" / "sanity_griffinlim.wav"
    out.parent.mkdir(exist_ok=True)
    # Normalize peak
    if np.max(np.abs(y)) > 0:
        y = 0.9 * y / np.max(np.abs(y))
    sf.write(str(out), y, SR)
    print(f"wrote {out} ({len(y)} samples, {len(y)/SR*1000:.0f} ms)")
