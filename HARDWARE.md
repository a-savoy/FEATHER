# FEATHER — Hardware & Wiring

FEATHER assumes a specific, deliberately simple rig. The whole point is that the
sound-card output is recorded **on the same acquisition board and clock as the
neural data**, so stimulus timing is exact with no drift and no unknown offset.

---

## Devices

| Device | Role |
|--------|------|
| Neural probe | e.g. Cambridge NeuroTech **ASSY-107**, 128 ch (`H5-D` single shank double-sided, or `H15` two shanks). Maps provided: `probe_H5D.json`, `probe_H15.json`. |
| Headstage / amplifier | e.g. CN 107H → Open Ephys acquisition board. |
| Open Ephys acquisition board + I/O (ADC) board | Records neural channels **and** the ADC sync channels, one clock (commonly 30 kHz). |
| **Harp Sound Card** | Stores stimuli in memory and plays them on command over its COM serial port. Emits a digital "sound-playing" gate. |
| Analog amplifier + speaker | e.g. Bose amp → FS2SE. **Use an analog amp/speaker** — a Bluetooth/DSP speaker adds variable latency and would turn a fixed offset into jitter. |
| (optional) Microphone at the animal | On a spare ADC to measure the true acoustic delay to the animal. |

---

## Wiring

```
                                   Harp Sound Card
                                   ┌───────────────┐
   COM port (USB, 1,000,000 baud) ─┤ serial        │
   play / stop / configure gate    │               │
                                    │  Left  RCA ───┼──► analog amp ──► speaker     (what the animal hears)
                                    │  Right RCA ───┼──► I/O board IN 2 ─► ADC2      (audio copy, in the recording)
                                    │  OUT1 (DO0) ──┼──► I/O board IN 1 ─► ADC1      (digital gate: HIGH while a sound plays)
                                    └───────────────┘
   probe ──► headstage ──► acquisition board ──► neural channels
   (optional) mic at animal ──────────────────► I/O board IN 3 ─► ADC3
```

### The ADC channel map (what FEATHER expects)

| Channel | Signal | Used for |
|---------|--------|----------|
| **ADC1** | Harp OUT1 digital gate. **HIGH while a sound plays; the rising edge = stimulus onset.** | Sample-accurate onset time. |
| **ADC2** | Copy of the played audio (Harp Right RCA). | Confirms *which* sound played and *exactly when* (cross-correlation against the stimulus WAVs). |
| **ADC3** | *(optional)* microphone at the animal. | Measures the speaker→air acoustic delay. Set `MIC_CHANNEL = None` in `playback_gui.py` to disable. |

These channel numbers are wiring conventions used throughout FEATHER. If you wire
differently, change `MIC_CHANNEL` and the `ADC1`/`ADC2` references (they are named
constants / clearly-labelled lookups in the code).

---

## The gate is a firmware feature (set once)

The gate on ADC1 is produced by the Harp card itself: digital output **DO0** is
configured to **"HighWhenSound"** so it is automatically HIGH for the exact
duration of every playback. FEATHER sets this once at the start of a run
(`GATE_OUTPUT = 0` in `playback_gui.py` = DO0). It is not toggled by software
during playback, so trigger latency and jitter cannot corrupt the recorded onset —
the gate and audio record the *true* output time on the ephys clock.

---

## Sound card format

The Harp card stores **stereo int32 at 96 kHz** (or 192 kHz). FEATHER's uploader
(`harp_upload.py`) resamples your WAVs and duplicates mono to both channels
(Left → speaker, Right → ADC2 copy) automatically. Your source stimuli can be any
uncompressed PCM WAV (e.g. 44.1 kHz mono); keep them all the same sample rate.

---

## Serial control summary (`harp_control.py`)

- Port: **COM4** by default, **1,000,000 baud**. Change `PORT` in `harp_control.py`
  if your card enumerates on a different COM number.
- `WhoAmI` (register 0) returns **1280** for the Sound Card — used as a safe
  reachability check (`python harp_control.py`).
- `PlaySoundOrFrequency` (reg 32) plays a stored index (**audible** — gate the
  animal's welfare before firing). `Stop` (reg 33). `ConfigureDO0/1/2`
  (reg 65/66/67), mode 2 = HighWhenSound.

**Only one program may hold COM4 at a time.** Run a single FEATHER window; if the
port gets stuck ("Access is denied"), unplug/replug the card's USB (see
INSTALL.md → Troubleshooting).

---

## Why this is sample-accurate

- **One clock.** Neural data and the ADC sync channels are digitized together, so
  there is no clock drift between "when the sound played" and "when the neuron
  fired."
- **Recorded, not commanded, timing.** The onset is read from the recorded gate
  edge, not from when software *asked* for playback — so USB/serial latency is
  irrelevant.
- **Measured result on the reference rig:** the recorded audio (ADC2) sits on the
  ADC1 marker to within one sample (~0.03 ms), and the gate width matches each
  stimulus's true duration with sub-millisecond jitter. Any fixed electrical vs.
  acoustic offset (speaker → air → animal, typically tens of ms) is measured
  separately by the optional ADC3 microphone.
