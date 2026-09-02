"""test_song_features.py — ground-truth checks for the acoustic features.

Every test here uses a signal whose answer is known by construction, so a failure
means the measurement is wrong rather than that the bird sounded different. These
catch sign errors, off-by-one framing, and band-limit mistakes.

They do NOT establish agreement with Sound Analysis Pro. That needs SAP-exported
tables for the same files; see README (Sound Bench) for the cross-check procedure.

Run:  python test_song_features.py       (or: pytest test_song_features.py)
"""
import numpy as np

import song_features as sfx

FS = 44100


# ---------------------------------------------------------------------------
# signal generators
# ---------------------------------------------------------------------------
def tone(freq, dur=1.0, amp=0.5, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def noise(dur=1.0, amp=0.2, fs=FS, seed=0):
    return np.random.default_rng(seed).normal(0, amp, int(dur * fs))


def sweep(f0, f1, dur=1.0, amp=0.5, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * dur)))


def harmonic_stack(f0, n_harm=8, dur=1.0, amp=0.5, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, n_harm + 1))
    return amp * y / np.max(np.abs(y))


def bursts(freq, n, on_ms, off_ms, fs=FS, amp=0.5):
    """n tone bursts with cosine edges, separated by silence."""
    on, off = int(on_ms * 1e-3 * fs), int(off_ms * 1e-3 * fs)
    ramp = max(1, int(0.003 * fs))
    env = np.ones(on)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    piece = tone(freq, on / fs, amp, fs) * env
    out = [np.zeros(off)]
    for _ in range(n):
        out += [piece, np.zeros(off)]
    return np.concatenate(out)


def mid(feat, col, lo=0.25, hi=0.75):
    """Middle of a steady signal, avoiding onset/offset frames."""
    n = len(feat)
    return feat[col].to_numpy()[int(n * lo):int(n * hi)]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_mean_frequency_finds_a_tone():
    feat, sp = sfx.measure(tone(2000), FS)
    got = np.median(mid(feat, "mean_frequency"))
    bin_hz = sp["freqs"][1] - sp["freqs"][0]
    assert abs(got - 2000) < 2 * bin_hz, f"2 kHz tone measured {got:.0f} Hz (bin {bin_hz:.0f} Hz)"


def test_entropy_noise_versus_tone():
    e_noise = np.median(mid(sfx.measure(noise(), FS)[0], "entropy"))
    e_tone = np.median(mid(sfx.measure(tone(3000), FS)[0], "entropy"))
    assert e_noise > -1.0, f"white noise entropy should approach 0, got {e_noise:.2f}"
    assert e_tone < -3.0, f"pure tone entropy should be strongly negative, got {e_tone:.2f}"
    assert e_noise - e_tone > 2.0


def test_entropy_is_gain_invariant():
    """Amplitude must not change entropy — the bird moving in the cage cannot."""
    quiet = np.median(mid(sfx.measure(tone(3000, amp=0.05), FS)[0], "entropy"))
    loud = np.median(mid(sfx.measure(tone(3000, amp=0.50), FS)[0], "entropy"))
    assert abs(quiet - loud) < 0.5, f"entropy moved {abs(quiet - loud):.2f} with gain"


def test_amplitude_tracks_level_in_db():
    a1 = np.median(mid(sfx.measure(tone(3000, amp=0.05), FS)[0], "amplitude"))
    a2 = np.median(mid(sfx.measure(tone(3000, amp=0.50), FS)[0], "amplitude"))
    assert abs((a2 - a1) - 20.0) < 1.5, f"10x amplitude should be +20 dB, got {a2 - a1:.1f}"


def test_fm_rises_with_modulation_rate():
    """FM must be ~0 for a flat tone and grow with sweep rate.

    Rate matters more than range: a 2-8 kHz sweep spread over a full second moves
    only ~56 Hz inside one 9.27 ms window, which really is almost no modulation.
    Birdsong modulates far faster, so the sweeps here are birdsong-rate.
    """
    fm_flat = np.median(mid(sfx.measure(tone(4000), FS)[0], "fm"))
    assert fm_flat < 2.0, f"flat tone should have ~0 FM, got {fm_flat:.1f} deg"

    fms = [np.median(mid(sfx.measure(sweep(2000, 8000, dur), FS)[0], "fm"))
           for dur in (0.25, 0.10, 0.05)]          # 24, 60, 120 kHz/s
    assert fms[0] > fm_flat + 10, f"a 24 kHz/s sweep should register: {fms[0]:.1f} deg"
    assert fms == sorted(fms), f"FM should rise with sweep rate, got {np.round(fms, 1)}"
    assert fms[-1] < 90.0, "FM is an angle and cannot exceed 90 degrees"


def test_pitch_recovers_the_fundamental():
    feat, _ = sfx.measure(harmonic_stack(700), FS)
    got = np.median(mid(feat, "pitch"))
    assert abs(got - 700) < 70, f"F0 = 700 Hz stack measured {got:.0f} Hz"


def test_goodness_of_pitch_ranks_harmonic_above_noise():
    g_harm = np.median(mid(sfx.measure(harmonic_stack(700), FS)[0], "goodness"))
    g_noise = np.median(mid(sfx.measure(noise(), FS)[0], "goodness"))
    assert g_harm > g_noise, f"harmonic {g_harm:.3f} should exceed noise {g_noise:.3f}"


def test_am_is_positive_at_onset_and_negative_at_offset():
    x = bursts(4000, 1, on_ms=200, off_ms=200)
    feat, _ = sfx.measure(x, FS)
    amp = feat["amplitude"].to_numpy()
    am = feat["am"].to_numpy()
    loud = amp > amp.max() - 20
    on = np.argmax(loud)                       # first frame over threshold
    off = len(loud) - 1 - np.argmax(loud[::-1])
    assert am[on:on + 12].max() > 0, "AM should rise at note onset"
    assert am[max(0, off - 12):off + 1].min() < 0, "AM should fall at note offset"


def test_segmentation_counts_and_times_bursts():
    n, on_ms, off_ms = 5, 90.0, 120.0
    feat, syl, sp = sfx.analyze(bursts(4000, n, on_ms, off_ms), FS)
    assert len(syl) == n, f"expected {n} syllables, got {len(syl)}"
    dur = syl["duration_ms"].to_numpy()
    assert np.all(np.abs(dur - on_ms) < 25), f"durations {np.round(dur, 1)} vs {on_ms} ms"
    gaps = syl["start_s"].to_numpy()[1:] - syl["end_s"].to_numpy()[:-1]
    assert np.all(np.abs(gaps * 1e3 - off_ms) < 30), f"gaps {np.round(gaps * 1e3, 1)} ms"


def test_duration_limits_reject_clicks_and_drones():
    x = np.concatenate([bursts(4000, 2, 90, 150), bursts(4000, 2, 3, 150)])   # + clicks
    _, syl, _ = sfx.analyze(x, FS, segment_params=dict(min_duration_ms=15))
    assert len(syl) == 2, f"3 ms clicks should be rejected, got {len(syl)} units"


def test_highpass_noise_detector_rejects_cage_rumble():
    """A 300 Hz thump is loud but is not a vocalization."""
    rumble = bursts(300, 3, 120, 150)
    calls = bursts(5000, 3, 120, 150)
    x = np.concatenate([rumble, calls])
    # Band must reach low enough to *see* the rumble, or it is filtered not rejected.
    sp_p = dict(f_low=100.0)
    _, syl, _ = sfx.analyze(x, FS, spec_params=sp_p,
                            segment_params=dict(noise_highpass_hz=1000.0))
    assert len(syl) == 3, f"expected only the 3 vocal bursts, got {len(syl)}"
    assert syl["mean_frequency_mean"].min() > 1000

    _, syl_off, _ = sfx.analyze(x, FS, spec_params=sp_p,
                                segment_params=dict(noise_highpass_hz=0))
    assert len(syl_off) == 6, "with the detector off, all 6 units should pass"


def test_bouts_group_syllables():
    x = np.concatenate([bursts(4000, 3, 90, 60), np.zeros(int(0.6 * FS)),
                        bursts(4000, 2, 90, 60)])
    _, syl, _ = sfx.analyze(x, FS)
    b = sfx.bouts(syl)
    assert len(b) == 2, f"expected 2 bouts, got {len(b)}"
    assert list(b["n_syllables"]) == [3, 2]


def test_empty_and_tiny_input_do_not_crash():
    for x in (np.zeros(0), np.zeros(10), np.zeros(FS // 100)):
        feat, syl, sp = sfx.analyze(x, FS)
        assert len(syl) == 0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
