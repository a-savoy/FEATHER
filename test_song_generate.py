"""test_song_generate.py — the call generator, and how it reaches the experiment.

These check the FEATHER side: that settings actually change the audio, that a saved
stimulus is a normal WAV the Design tab can use, and that supernormal settings are
flagged and recorded rather than silently clipped.

The VAE itself is upstream (see callgen/SOURCE.md) and is not re-tested here.

Run:  python test_song_generate.py
"""
import os
import json
import tempfile

import numpy as np
import soundfile as sf

import playback_tools as pt
import song_generate as sg

SKIP = None
ok, msg = sg.available()
if not ok:
    SKIP = msg


def test_generator_is_installed():
    assert SKIP is None, SKIP


def test_renders_audio_at_the_right_rate():
    y, specs = sg.render(dict(calls_per_bout=3, n_bouts=1, gl_iters=20, seed=1))
    assert y.dtype == np.float32 and y.ndim == 1
    assert len(specs) == 3, f"3 calls requested, {len(specs)} spectrograms"
    assert 0 < np.abs(y).max() <= 1.0
    assert 0.2 < len(y) / sg.SR < 5.0, f"unexpected length {len(y) / sg.SR:.2f} s"


def test_same_seed_is_reproducible_and_different_seeds_are_not():
    a, _ = sg.render(dict(calls_per_bout=2, n_bouts=1, gl_iters=20, seed=7))
    b, _ = sg.render(dict(calls_per_bout=2, n_bouts=1, gl_iters=20, seed=7))
    c, _ = sg.render(dict(calls_per_bout=2, n_bouts=1, gl_iters=20, seed=8))
    assert np.array_equal(a, b), "same settings and seed must give identical audio"
    assert not (len(a) == len(c) and np.array_equal(a, c)), "a new seed should differ"


def test_bout_structure_follows_the_settings():
    """More calls and longer gaps must make a longer file."""
    short, _ = sg.render(dict(calls_per_bout=3, n_bouts=1, ici_ms=100, gl_iters=20, seed=2))
    many, _ = sg.render(dict(calls_per_bout=9, n_bouts=1, ici_ms=100, gl_iters=20, seed=2))
    spaced, _ = sg.render(dict(calls_per_bout=3, n_bouts=1, ici_ms=300, gl_iters=20, seed=2))
    assert len(many) > len(short) * 2, "9 calls should be much longer than 3"
    assert len(spaced) > len(short), "wider gaps should lengthen the bout"


def test_feathers_own_analysis_finds_the_calls_it_asked_for():
    """The generator and the analysis code should agree on how many calls there are."""
    n = 6
    y, _ = sg.render(dict(calls_per_bout=n, n_bouts=1, duration_ms=100,
                          ici_ms=150, gl_iters=20, seed=3))
    d = sg.describe(y)
    assert d["calls"] == n, f"asked for {n} calls, analysis found {d['calls']}"
    assert 50 < d["median_duration_ms"] < 200


def test_supernormal_settings_are_detected():
    assert sg.is_supernormal("age_days", 25), "day 25 is outside the 4-15 training range"
    assert not sg.is_supernormal("age_days", 10)
    assert sg.is_supernormal("f0_khz", 8.5), "8.5 kHz is above the species-typical F0"
    assert not sg.is_supernormal("f0_khz", 3.9)
    flags = sg.supernormal_settings(dict(age_days=25, f0_khz=3.9, noisiness=0.5))
    assert flags == ["age_days"], flags


def test_brightness_slider_is_monotonic():
    """Centroid is the dependable spectral axis: raising it raises the measured centre.

    Note this is NOT true of the F0 slider — see test_f0_changes_the_sound below.
    """
    centres = []
    for v in (2.0, 3.0, 4.0, 5.0):
        y, _ = sg.render(dict(centroid_khz=v, calls_per_bout=2, n_bouts=1,
                              gl_iters=20, seed=5))
        centres.append(sg.describe(y)["median_centre_hz"])
    assert centres == sorted(centres), f"brightness should rise monotonically: {centres}"
    assert centres[-1] - centres[0] > 1000, f"and by a useful amount: {centres}"


def test_f0_changes_the_sound():
    """The F0 slider does change the call — but not monotonically in measured centre.

    The engine moves F0 along a direction orthogonalized against centroid, and in
    practice the measured spectral centre falls as F0 rises from 2 to 5 kHz, then
    flattens. So this asserts only that the sound really changes; use the brightness
    slider when you need a graded spectral ladder.
    """
    a, _ = sg.render(dict(f0_khz=2.0, calls_per_bout=2, n_bouts=1, gl_iters=20, seed=5))
    b, _ = sg.render(dict(f0_khz=6.0, calls_per_bout=2, n_bouts=1, gl_iters=20, seed=5))
    da, db = sg.describe(a), sg.describe(b)
    assert abs(da["median_centre_hz"] - db["median_centre_hz"]) > 500, (
        f"F0 2 kHz vs 6 kHz should sound clearly different: "
        f"{da['median_centre_hz']:.0f} vs {db['median_centre_hz']:.0f} Hz")


def test_extreme_settings_still_produce_playable_audio():
    """Supernormal values are passed through unclipped, so check they stay safe to play."""
    for params in (dict(age_days=20.0), dict(f0_khz=9.0), dict(noisiness=1.5),
                   dict(sex=-0.5), dict(duration_ms=300.0)):
        params.update(calls_per_bout=2, n_bouts=1, gl_iters=20, seed=5)
        y, _ = sg.render(params)
        assert np.all(np.isfinite(y)), f"{params} produced non-finite samples"
        assert np.abs(y).max() > 1e-4, f"{params} produced silence"
        assert np.abs(y).max() <= 1.0, f"{params} would clip the output"


def test_saved_stimulus_is_usable_by_the_design_tab():
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "stimuli")
        params = dict(sg.defaults()); params.update(calls_per_bout=4, n_bouts=1, gl_iters=20)
        y, _ = sg.render(params)
        out = sg.save_stimulus(y, dest, params)
        info = sf.info(out)
        assert info.channels == 1 and info.subtype == "PCM_16"
        table = pt.scan_stimuli(dest)
        assert pt.check_uniform_sample_rate(table) == sg.SR
        assert len(table) == 1


def test_saving_matches_an_existing_folders_sample_rate():
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "stimuli")
        os.makedirs(dest)
        sf.write(os.path.join(dest, "existing.wav"), np.zeros(48000, np.float32), 48000)
        params = dict(sg.defaults()); params.update(calls_per_bout=2, n_bouts=1, gl_iters=20)
        y, _ = sg.render(params)
        out = sg.save_stimulus(y, dest, params)
        assert sf.info(out).samplerate == 48000, "must convert to the folder's rate"
        pt.check_uniform_sample_rate(pt.scan_stimuli(dest))


def test_provenance_records_every_parameter():
    import song_library as slib
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "stimuli")
        params = dict(sg.defaults())
        params.update(calls_per_bout=2, n_bouts=1, gl_iters=20, age_days=25.0, seed=42)
        y, _ = sg.render(params)
        sg.save_stimulus(y, dest, params, note="ladder step 3")
        prov = slib.provenance(dest)
        assert len(prov) == 1
        row = prov.iloc[0]
        assert row["source"] == "VAE generator (callgen)"
        saved = json.loads(row["parameters"])
        assert saved["age_days"] == 25.0 and saved["seed"] == 42
        assert "SUPERNORMAL" in row["note"] and "age_days" in row["note"]
        assert "ladder step 3" in row["note"]


def test_series_makes_one_stimulus_per_step():
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "stimuli")
        base = dict(sg.defaults()); base.update(calls_per_bout=2, n_bouts=1, gl_iters=20)
        outs = sg.series(base, "centroid_khz", [2.5, 4.0, 5.5], dest, name_prefix="ladder")
        assert len(outs) == 3 and len(set(outs)) == 3
        table = pt.scan_stimuli(dest)
        assert len(table) == 3
        pt.check_uniform_sample_rate(table)
        # the ladder must actually be graded, not three copies
        centres = [sg.describe(sf.read(o)[0])["median_centre_hz"] for o in outs]
        assert centres == sorted(centres), f"brightness ladder should rise: {np.round(centres)}"


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
