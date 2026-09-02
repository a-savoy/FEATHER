"""test_song_library.py — scanning, caching, and stimulus export.

The export tests matter most: they write into the folder Design Experiment reads, so
a clip that lands at the wrong sample rate would break the play order downstream.

Run:  python test_song_library.py
"""
import os
import time
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf

import playback_tools as pt
import song_library as slib

FS = 44100


def call_bout(n=5, freq=5000, fs=FS):
    parts = [np.zeros(int(0.1 * fs))]
    for _ in range(n):
        t = np.arange(int(0.09 * fs)) / fs
        env = np.ones(len(t))
        r = max(1, int(0.004 * fs))
        env[:r], env[-r:] = np.linspace(0, 1, r), np.linspace(1, 0, r)
        parts += [0.5 * np.sin(2 * np.pi * freq * t) * env, np.zeros(int(0.08 * fs))]
    return np.concatenate(parts)


def make_tree(root):
    """A song_data-shaped tree: bird / day / files, plus a rejected one."""
    day = os.path.join(root, "A2N3", "2026-09-02")
    os.makedirs(os.path.join(day, "rejected"), exist_ok=True)
    p1 = os.path.join(day, "A2N3_20260902_101500_000_ch1.wav")
    p2 = os.path.join(day, "A2N3_20260902_101800_000_ch1.wav")
    p3 = os.path.join(day, "rejected", "A2N3_20260902_102000_000_ch1.wav")
    sf.write(p1, call_bout(5).astype(np.float32), FS, subtype="PCM_16")
    sf.write(p2, call_bout(3).astype(np.float32), FS, subtype="PCM_16")
    sf.write(p3, np.zeros(FS // 2, dtype=np.float32), FS, subtype="PCM_16")
    return p1, p2, p3


# ---------------------------------------------------------------------------
def test_scan_reads_bird_and_day_from_the_layout():
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp)
        t = slib.scan_folder(tmp)
        assert len(t) == 2, f"rejected/ should be skipped by default, got {len(t)} files"
        assert set(t["bird"]) == {"A2N3"}
        assert set(t["day"]) == {"2026-09-02"}
        assert len(slib.scan_folder(tmp, include_rejected=True)) == 3


def test_cache_returns_the_same_answer_and_is_faster():
    with tempfile.TemporaryDirectory() as tmp:
        p1, _, _ = make_tree(tmp)
        lib = slib.Library(os.path.join(tmp, slib.CACHE_NAME))
        t0 = time.perf_counter(); a = lib.analyze(p1); cold = time.perf_counter() - t0
        t0 = time.perf_counter(); b = lib.analyze(p1); warm = time.perf_counter() - t0
        assert len(a) == 5, f"expected 5 calls, got {len(a)}"
        pd.testing.assert_frame_equal(a, b)
        assert warm < cold, f"cached read ({warm * 1e3:.1f} ms) should beat cold ({cold * 1e3:.1f} ms)"
        lib.close()


def test_cache_invalidates_when_the_file_changes():
    with tempfile.TemporaryDirectory() as tmp:
        p1, _, _ = make_tree(tmp)
        lib = slib.Library(os.path.join(tmp, slib.CACHE_NAME))
        assert len(lib.analyze(p1)) == 5
        sf.write(p1, call_bout(2).astype(np.float32), FS, subtype="PCM_16")
        assert len(lib.analyze(p1)) == 2, "a rewritten file must be re-analysed"
        lib.close()


def test_cache_invalidates_when_settings_change():
    with tempfile.TemporaryDirectory() as tmp:
        p1, _, _ = make_tree(tmp)
        lib = slib.Library(os.path.join(tmp, slib.CACHE_NAME))
        loose = lib.analyze(p1, segment_params=dict(min_duration_ms=15))
        strict = lib.analyze(p1, segment_params=dict(min_duration_ms=200))
        assert len(loose) == 5 and len(strict) == 0, (
            f"different thresholds must give different results: {len(loose)} vs {len(strict)}")
        lib.close()


def test_analyze_many_and_summary():
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp)
        lib = slib.Library(os.path.join(tmp, slib.CACHE_NAME))
        t = slib.scan_folder(tmp)
        table = lib.analyze_many(list(t["path"]))
        assert set(table["file"].unique()) == set(t["file"])
        s = lib.summary(table)
        assert list(s["syllables"]) == [5, 3] or list(s["syllables"]) == [3, 5]
        lib.close()


def test_export_matches_the_folders_existing_sample_rate():
    """Design Experiment refuses a folder of mixed rates, so export must convert."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "rec_48k.wav")
        sf.write(src, call_bout(3, fs=48000).astype(np.float32), 48000, subtype="PCM_16")
        dest = os.path.join(tmp, "stimuli")
        os.makedirs(dest)
        sf.write(os.path.join(dest, "existing.wav"),
                 np.zeros(FS, dtype=np.float32), FS, subtype="PCM_16")

        out = slib.export_stimulus(src, dest, start_s=0.1, end_s=0.6)
        assert sf.info(out).samplerate == FS, "clip should be resampled to the folder rate"
        rate = pt.check_uniform_sample_rate(pt.scan_stimuli(dest))
        assert rate == FS


def test_export_writes_mono_16bit_of_the_right_length():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "rec.wav")
        sf.write(src, call_bout(5).astype(np.float32), FS, subtype="PCM_16")
        dest = os.path.join(tmp, "stimuli")
        out = slib.export_stimulus(src, dest, start_s=0.20, end_s=0.95)
        info = sf.info(out)
        assert info.channels == 1 and info.subtype == "PCM_16"
        assert abs(info.duration - 0.75) < 0.01, f"expected 0.75 s, got {info.duration:.3f}"


def test_export_fades_so_the_clip_does_not_click():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "rec.wav")
        # Cut straight through a loud tone: without a fade this starts at full amplitude.
        t = np.arange(FS) / FS
        sf.write(src, (0.8 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32), FS,
                 subtype="PCM_16")
        dest = os.path.join(tmp, "stimuli")
        out = slib.export_stimulus(src, dest, start_s=0.2, end_s=0.8, fade_ms=5)
        x, _ = sf.read(out)
        assert abs(x[0]) < 0.05 and abs(x[-1]) < 0.05, "clip edges should be faded"
        assert np.abs(x).max() > 0.5, "but the body of the clip should be intact"


def test_export_records_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "A2N3_call.wav")
        sf.write(src, call_bout(5).astype(np.float32), FS, subtype="PCM_16")
        dest = os.path.join(tmp, "stimuli")
        slib.export_stimulus(src, dest, start_s=0.10, end_s=0.55, note="day 11")
        slib.export_stimulus(src, dest, start_s=0.60, end_s=1.05, note="day 11")
        prov = slib.provenance(dest)
        assert len(prov) == 2
        assert set(prov["source"]) == {os.path.abspath(src)}
        assert "start_s" in prov and "end_s" in prov and "exported" in prov
        assert prov["note"].iloc[0] == "day 11"


def test_export_never_overwrites_an_existing_stimulus():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "rec.wav")
        sf.write(src, call_bout(5).astype(np.float32), FS, subtype="PCM_16")
        dest = os.path.join(tmp, "stimuli")
        a = slib.export_stimulus(src, dest, start_s=0.1, end_s=0.5, name="call")
        b = slib.export_stimulus(src, dest, start_s=0.1, end_s=0.5, name="call")
        assert a != b and os.path.exists(a) and os.path.exists(b)


def test_mixed_rate_folder_is_reported_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "stimuli")
        os.makedirs(dest)
        sf.write(os.path.join(dest, "a.wav"), np.zeros(1000, np.float32), 44100)
        sf.write(os.path.join(dest, "b.wav"), np.zeros(1000, np.float32), 48000)
        try:
            slib.folder_sample_rate(dest)
        except ValueError as e:
            assert "mixed sample rates" in str(e)
        else:
            raise AssertionError("a mixed-rate folder should raise")


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
