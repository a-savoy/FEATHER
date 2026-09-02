"""song_recorder.py — sound-activated multi-channel recording of bird vocalizations.

FEATHER's replacement for the Sound Analysis Pro Recorder. Each channel is one bird
in one box, watched independently. Sound above threshold opens a recording, fed from a
ring buffer so the onset is never clipped; silence closes it.

Two-stage keep/discard, as SAP does it:

  Stage 1 (live, here)  amplitude over background opens a session; it closes after
                        `hold_s` of silence; the session is kept only if the envelope
                        rose above threshold at least `min_events` separate times, so
                        a single door slam is discarded.
  Stage 2 (after each   the recorded audio is measured and segmented, then accepted on
   file, song_features) its STRUCTURE: enough syllables, of plausible duration, forming
                        a long enough bout, and not mostly low-frequency cage noise.

One deliberate difference from SAP: rejected files are MOVED, not deleted. They go to
a `rejected/` folder with the failing criterion recorded in events.csv, so thresholds
can be audited and re-tuned instead of silently losing data.

Audio callbacks only ever hand blocks to a queue — all file and analysis work happens
on a worker thread, so a slow disk can never glitch the stream.
"""
from __future__ import annotations

import os
import csv
import time
import queue
import shutil
import threading
from collections import deque

import numpy as np
import soundfile as sf

import song_features as sfx

TRIGGER_DEFAULTS = dict(
    threshold_db=-40.0,       # envelope level that opens a recording (dBFS)
    hold_s=1.0,               # silence this long closes it (SAP uses 1 s)
    prebuffer_s=2.0,          # audio kept from before the trigger
    min_events=2,             # separate threshold crossings needed to keep the file
    min_duration_s=0.20,
    max_duration_s=60.0,
    highpass_hz=400.0,        # envelope is measured above this, so rumble cannot trigger
    highpass_order=4,         # steep enough that loud cage rumble stays under threshold
)

# Stage 2. These are set for chick begging calls, not adult song: song-tuned bout
# criteria would discard short repetitive begging bursts.
ACCEPT_DEFAULTS = dict(
    min_syllables=3,
    min_bout_ms=150.0,
    enabled=True,
)

BLOCK = 2048


def list_devices():
    """Input devices as (index, label, max_input_channels, default_samplerate)."""
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], int(d["max_input_channels"]),
                        float(d["default_samplerate"])))
    return out


def _db(x):
    """RMS level of a block in dBFS."""
    r = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return 20.0 * np.log10(max(r, 1e-9))


class _ChannelState:
    """Stage-1 state machine for one channel."""

    def __init__(self, name, fs, trig):
        self.name = name
        self.fs = fs
        self.trig = trig
        self.pre = deque(maxlen=max(1, int(trig["prebuffer_s"] * fs / BLOCK)))
        self.writer = None
        self.path = None
        self.started = None
        self.frames = 0
        self.events = 0
        self.above = False
        self.silent_for = 0.0
        self.peak_db = -120.0
        self.level_db = -120.0


class Recorder:
    """Sound-activated recorder. Start it, poll levels(), stop it.

    on_event(dict) is called from the worker thread for each finished file, with keys
    file, bird, channel, started, duration_s, peak_db, accepted, reason.
    """

    def __init__(self, outdir, device=None, channels=1, fs=44100.0,
                 bird_names=None, trigger=None, accept=None, continuous=False,
                 continuous_s=60.0, on_event=None, spec_params=None, segment_params=None):
        self.outdir = outdir
        self.device = device
        self.n_ch = int(channels)
        self.fs = float(fs)
        self.bird_names = list(bird_names or [f"ch{i + 1}" for i in range(self.n_ch)])
        self.trig = dict(TRIGGER_DEFAULTS, **(trigger or {}))
        self.accept = dict(ACCEPT_DEFAULTS, **(accept or {}))
        self.continuous = bool(continuous)
        self.continuous_s = float(continuous_s)
        self.on_event = on_event
        self.spec_params = spec_params or {}
        self.segment_params = segment_params or {}

        self.q = queue.Queue(maxsize=512)
        self.monitor = queue.Queue(maxsize=64)     # decimated audio for the live display
        self._stop = threading.Event()
        self._stream = None
        self._worker = None
        self._hp_state = None
        self.error = None
        self.n_kept = 0
        self.n_rejected = 0
        self.dropped_blocks = 0
        self.chans = [_ChannelState(n, self.fs, self.trig)
                      for n in self.bird_names[:self.n_ch]]

    # -- public ------------------------------------------------------------
    def start(self):
        import sounddevice as sd
        os.makedirs(self.outdir, exist_ok=True)
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._stream = sd.InputStream(
            samplerate=self.fs, device=self.device, channels=self.n_ch,
            blocksize=BLOCK, dtype="float32", callback=self._callback)
        self._stream.start()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def levels(self):
        """Current level per channel, dBFS — for the meters."""
        return [c.level_db for c in self.chans]

    def recording_flags(self):
        return [c.writer is not None for c in self.chans]

    # -- audio thread ------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        # Never block, never touch the disk here.
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            self.dropped_blocks += 1

    # -- worker thread -----------------------------------------------------
    def _run(self):
        try:
            while not self._stop.is_set() or not self.q.empty():
                try:
                    blk = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._process(blk)
            for ch in self.chans:                    # flush on stop
                if ch.writer is not None:
                    self._close(ch)
        except Exception as e:                       # keep the GUI informed
            self.error = f"{type(e).__name__}: {e}"

    def _envelope(self, blk):
        """Per-channel level in dB, measured above the highpass so rumble is ignored."""
        if self.trig["highpass_hz"] and blk.shape[0] > 8:
            from scipy.signal import butter, sosfilt
            if self._hp_state is None:
                self._sos = butter(int(self.trig["highpass_order"]),
                                   self.trig["highpass_hz"] / (self.fs / 2),
                                   btype="high", output="sos")
                # Start from rest. sosfilt_zi() returns step-response conditions, which
                # would make the filter ring on the first block and trigger on silence.
                self._hp_state = np.zeros((self._sos.shape[0], 2, blk.shape[1]))
            filt, self._hp_state = sosfilt(self._sos, blk, axis=0, zi=self._hp_state)
        else:
            filt = blk
        return [_db(filt[:, i]) for i in range(blk.shape[1])], filt

    def _process(self, blk):
        levels, _ = self._envelope(blk)
        dur = blk.shape[0] / self.fs
        try:
            self.monitor.put_nowait(blk[:, 0].copy())
        except queue.Full:
            pass

        for i, ch in enumerate(self.chans):
            x = blk[:, i]
            ch.level_db = levels[i]
            loud = levels[i] > self.trig["threshold_db"]

            if self.continuous:
                if ch.writer is None:
                    self._open(ch, i)
                ch.writer.write(x)
                ch.frames += len(x)
                ch.peak_db = max(ch.peak_db, levels[i])
                if ch.frames / self.fs >= self.continuous_s:
                    self._close(ch)
                continue

            if ch.writer is None:
                ch.pre.append(x.copy())
                if loud:
                    self._open(ch, i)
                    for b in ch.pre:                 # the pre-trigger audio
                        ch.writer.write(b)
                        ch.frames += len(b)
                    ch.pre.clear()
                    ch.events = 1
                    ch.above = True
            else:
                ch.writer.write(x)
                ch.frames += len(x)
                ch.peak_db = max(ch.peak_db, levels[i])
                if loud:
                    if not ch.above:                 # a new excursion above threshold
                        ch.events += 1
                    ch.above = True
                    ch.silent_for = 0.0
                else:
                    ch.above = False
                    ch.silent_for += dur
                too_long = ch.frames / self.fs >= self.trig["max_duration_s"]
                if ch.silent_for >= self.trig["hold_s"] or too_long:
                    self._close(ch)

    # -- files -------------------------------------------------------------
    def _day_dir(self, bird, when):
        d = os.path.join(self.outdir, bird, time.strftime("%Y-%m-%d", time.localtime(when)))
        os.makedirs(d, exist_ok=True)
        return d

    def _open(self, ch, idx):
        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int(now % 1 * 1000):03d}"
        d = self._day_dir(ch.name, now)
        ch.path = os.path.join(d, f"{ch.name}_{stamp}_ch{idx + 1}.wav")
        ch.writer = sf.SoundFile(ch.path, mode="w", samplerate=int(self.fs),
                                 channels=1, subtype="PCM_16")
        ch.started, ch.frames, ch.events, ch.silent_for = now, 0, 0, 0.0
        ch.peak_db = -120.0

    def _close(self, ch):
        path, started, frames = ch.path, ch.started, ch.frames
        try:
            ch.writer.close()
        except Exception:
            pass
        ch.writer, ch.path = None, None
        ch.silent_for, ch.above = 0.0, False
        if not path or not os.path.exists(path):
            return
        dur = frames / self.fs

        accepted, reason = True, ""
        if not self.continuous:
            if dur < self.trig["min_duration_s"]:
                accepted, reason = False, f"too short ({dur * 1e3:.0f} ms)"
            elif ch.events < self.trig["min_events"]:
                accepted, reason = False, f"only {ch.events} threshold crossing(s)"
            elif self.accept["enabled"]:
                accepted, reason = self._stage2(path)

        row = dict(file=os.path.basename(path), bird=ch.name,
                   started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                   duration_s=round(dur, 3), peak_db=round(ch.peak_db, 1),
                   events=ch.events, accepted=accepted, reason=reason)

        if not accepted:
            rej = os.path.join(os.path.dirname(path), "rejected")
            os.makedirs(rej, exist_ok=True)
            dest = os.path.join(rej, os.path.basename(path))
            try:
                shutil.move(path, dest)
            except Exception:
                pass
            self.n_rejected += 1
        else:
            self.n_kept += 1

        self._log(os.path.dirname(path), row)
        if self.on_event:
            try:
                self.on_event(row)
            except Exception:
                pass

    def _stage2(self, path):
        """Accept on the structure of what was recorded, not just its loudness."""
        try:
            x, fs = sf.read(path)
            feat, syl, _sp = sfx.analyze(x, fs, spec_params=self.spec_params,
                                         segment_params=self.segment_params)
            if len(syl) < self.accept["min_syllables"]:
                return False, f"{len(syl)} syllable(s), need {self.accept['min_syllables']}"
            b = sfx.bouts(syl)
            longest = float(b["duration_ms"].max()) if len(b) else 0.0
            if longest < self.accept["min_bout_ms"]:
                return False, f"longest bout {longest:.0f} ms, need {self.accept['min_bout_ms']:.0f}"
            return True, ""
        except Exception as e:
            # Never lose audio because analysis failed — keep it and say why.
            return True, f"kept without check ({type(e).__name__})"

    def _log(self, day_dir, row):
        path = os.path.join(day_dir, "events.csv")
        new = not os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception:
            pass


def write_session(outdir, info):
    """Record how a recording session was configured, next to the audio."""
    import json
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "session.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
