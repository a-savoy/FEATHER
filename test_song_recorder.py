"""test_song_recorder.py — recorder logic without a microphone.

The audio callback is the only part that needs hardware, and it does nothing but hand
blocks to a queue. So these tests feed blocks straight into the worker's processing
step and check the decisions: when a file opens, when it closes, what gets kept, what
gets moved to rejected/, and whether the pre-buffer really saves the onset.

Run:  python test_song_recorder.py
"""
import os
import glob
import shutil
import tempfile

import numpy as np
import soundfile as sf

import song_recorder as srec

FS = 44100.0
B = srec.BLOCK


def feed(rec, x, n_ch=1):
    """Push a mono signal through the recorder as blocks, then flush."""
    for i in range(0, len(x) - B, B):
        blk = np.tile(x[i:i + B, None], (1, n_ch)).astype(np.float32)
        rec._process(blk)
    for ch in rec.chans:
        if ch.writer is not None:
            rec._close(ch)


def burst(freq, dur_s, amp=0.4, fs=FS):
    t = np.arange(int(dur_s * fs)) / fs
    env = np.ones(len(t))
    r = max(1, int(0.005 * fs))
    env[:r], env[-r:] = np.linspace(0, 1, r), np.linspace(1, 0, r)
    return (amp * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def silence(dur_s, fs=FS):
    return np.zeros(int(dur_s * fs), dtype=np.float32)


def begging_bout(n=5, fs=FS):
    """Repetitive calls, like a begging bout."""
    parts = [silence(0.3)]
    for _ in range(n):
        parts += [burst(5000, 0.09), silence(0.08)]
    parts.append(silence(1.5))
    return np.concatenate(parts)


def make(tmp, **kw):
    return srec.Recorder(outdir=tmp, channels=1, fs=FS, bird_names=["A2N3"], **kw)


def wavs(tmp):
    return sorted(glob.glob(os.path.join(tmp, "**", "*.wav"), recursive=True))


def kept(tmp):
    return [f for f in wavs(tmp) if "rejected" not in f]


def rejected(tmp):
    return [f for f in wavs(tmp) if "rejected" in f]


# ---------------------------------------------------------------------------
def test_silence_records_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp)
        feed(rec, silence(3.0))
        assert wavs(tmp) == [], "silence should not open a file"


def test_begging_bout_is_recorded_and_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp)
        feed(rec, begging_bout())
        k = kept(tmp)
        assert len(k) == 1, f"expected 1 kept file, got {len(k)} (rejected: {len(rejected(tmp))})"
        assert rec.n_kept == 1 and rec.n_rejected == 0


def test_single_thump_is_rejected_not_deleted():
    """One isolated event is not a vocalization — but we keep the evidence."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp)
        feed(rec, np.concatenate([silence(0.3), burst(5000, 0.05), silence(2.0)]))
        assert kept(tmp) == [], "a single short thump should not be kept"
        assert len(rejected(tmp)) == 1, "and it should still be on disk, in rejected/"
        assert rec.n_rejected == 1


def test_low_frequency_rumble_never_triggers():
    """The trigger listens above the highpass, so cage rumble cannot open a file."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp)
        feed(rec, np.concatenate([silence(0.3), burst(120, 1.0, amp=0.9), silence(1.5)]))
        assert wavs(tmp) == [], "120 Hz rumble should stay below the trigger"


def test_prebuffer_keeps_the_onset():
    """The saved file must start BEFORE the sound that triggered it."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp, trigger=dict(prebuffer_s=1.0))
        sound_s = 5 * 0.09 + 4 * 0.08
        feed(rec, begging_bout())
        k = kept(tmp)
        assert len(k) == 1
        info = sf.info(k[0])
        assert info.duration > sound_s + 0.5, (
            f"file is {info.duration:.2f} s; with a 1 s pre-buffer it should comfortably "
            f"exceed the {sound_s:.2f} s of sound")
        x, _ = sf.read(k[0])
        assert np.abs(x[:2000]).max() < 0.05, "the file should open on pre-trigger quiet"


def test_stage2_rejects_a_long_flat_drone():
    """Loud and repetitive is not enough — structure decides."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp, trigger=dict(hold_s=0.4))
        # A continuous 3 s tone: one long unit, no syllable structure.
        feed(rec, np.concatenate([silence(0.3), burst(5000, 3.0), silence(1.5)]))
        assert kept(tmp) == [], "a featureless drone should not pass stage 2"
        assert len(rejected(tmp)) == 1


def test_events_csv_records_the_decision():
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp)
        feed(rec, begging_bout())
        feed(rec, np.concatenate([silence(0.3), burst(5000, 0.05), silence(2.0)]))
        logs = glob.glob(os.path.join(tmp, "**", "events.csv"), recursive=True)
        assert logs, "events.csv should be written next to the audio"
        text = open(logs[0]).read()
        assert "accepted" in text and "reason" in text
        assert "True" in text and "False" in text, "both decisions should be logged"


def test_continuous_mode_splits_into_fixed_files():
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp, continuous=True, continuous_s=1.0)
        feed(rec, np.concatenate([burst(4000, 3.2)]))
        k = kept(tmp)
        assert len(k) >= 3, f"3.2 s at 1 s per file should give >=3 files, got {len(k)}"
        assert all(sf.info(f).duration <= 1.2 for f in k)


def test_files_land_in_bird_and_day_folders():
    with tempfile.TemporaryDirectory() as tmp:
        rec = make(tmp)
        feed(rec, begging_bout())
        f = kept(tmp)[0]
        rel = os.path.relpath(f, tmp).split(os.sep)
        assert rel[0] == "A2N3", f"first folder should be the bird, got {rel}"
        assert len(rel[1]) == 10 and rel[1][4] == "-", f"then a YYYY-MM-DD day, got {rel[1]}"
        assert rel[-1].startswith("A2N3_") and rel[-1].endswith("_ch1.wav")


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
