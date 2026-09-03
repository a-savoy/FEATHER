# F.E.A.T.H.E.R.

**Friendly Ephys Acquisition Terminal Helping Everyone Record**

FEATHER records bird vocalizations or synthesizes begging calls, then
runs a randomized **audio-playback experiment** during an extracellular
electrophysiology recording (or as a behavior-only session), then aids in analysis:
record the birds → segment the calls you want into stimuli (or generate new ones) →
design the play order →
play each stimulus through a Harp Sound Card → capture a sample-accurate onset marker
on an Open Ephys ADC channel → detect and identify every sound → spike-sort with
**Kilosort4** → view PSTHs → hand off to **Phy** or **Jupyter**. It is built for a neuroscience student with no coding experience: a
point-and-click GUI, plain-language labels, and sensible fixed defaults.

There are two equivalent front ends over the exact same tested code:

- **The app** — `playback_gui.py` (a desktop window, recommended for daily use).
- **The notebook** — `notebooks/FEATHER_notebook.ipynb` (a Jupyter alternative
  that does the identical pipeline, cell by cell, for people who prefer code).

---

## Why the timing is trustworthy

**The sound card's output is recorded on the
same acquisition board, on the same clock, as the neural data.**

- A digital "sound-playing" gate from the Harp card is wired to **ADC1**. Its
  rising edge is the stimulus onset — sample-accurate on the ephys clock.
- A copy of the audio is wired to **ADC2**, so the actual waveform lives in the
  recording and is used to confirm *which* sound played and *exactly when*.

Because everything shares one clock, there is no drift and no unknown offset. On
this rig the recorded audio sits on the timing marker to within a single sample
(~0.03 ms). See **HARDWARE.md** for the full wiring.

---

## What's in this package

| Path | What it is |
|------|-----------|
| `playback_gui.py` | **The FEATHER app.** Run this. Nine tabs: record vox → generate → sound library → design → run → analysis → saved files → Jupyter hand-off → about. |
| `notebooks/FEATHER_notebook.ipynb` | **Jupyter alternative to the app** — the whole pipeline as runnable cells. |
| `song_features.py` | **Sound Bench:** multi-taper spectrogram, spectral derivatives, and the seven acoustic features (amplitude, pitch, mean frequency, goodness of pitch, Wiener entropy, FM, AM); syllable and bout segmentation. |
| `song_recorder.py` | **Sound Bench:** sound-activated multi-channel recording with a pre-buffer and the two-stage keep/discard check. |
| `song_library.py` | **Sound Bench:** browse recordings, cache their measurements (SQLite), cut clips into stimuli with provenance. |
| `song_generate.py` | **Sound Bench:** synthetic begging calls from the trained VAE — controls, rendering, saving with full parameter provenance. |
| `callgen/` | The trained model, vendored unmodified from the Hugging Face Space (see `callgen/SOURCE.md`). |
| `song_tabs.py` | **Sound Bench:** the Record, Sound Library and Generate Calls tabs (mixed into the app). |
| `test_song_features.py`, `test_song_recorder.py`, `test_song_library.py`, `test_song_generate.py` | Ground-truth tests for the above — run them after any change (`python test_song_features.py`). |
| `playback_tools.py` | Builds the randomized play order / schedule. |
| `harp_control.py` | Drives the Harp Sound Card over its COM serial port (play / stop / configure the gate). |
| `harp_upload.py` | Uploads stimulus sounds into the card's memory. |
| `make_test_data.py` | Generates a **fake Open Ephys recording** so you can try the whole analysis with no hardware. |
| `build_probes.py` | Rebuilds the probe map files from the transcribed channel tables. |
| `probe_H5D.json`, `probe_H15.json` | Ready-to-use Kilosort/probeinterface maps for the two 128-ch Cambridge NeuroTech ASSY-107 probes. |
| `make_feather_icon.py` | Draws the zebra-finch feather icon. |
| `stimuli/` | **Sample stimuli** (4 zebra-finch begging calls, 5 s each, 44.1 kHz mono). Replace with your own WAVs. |
| `configs/` | Example saved settings. |
| `requirements.txt`, `environment-feather.yml` | The main FEATHER + Kilosort4 environment. |
| `requirements-phy2.txt`, `environment-phy2.yml` | The separate Phy 2 environment. |
| `run_feather.bat` | Convenience launcher (activates the env and opens the app). |
| `INSTALL.md` | Full setup: environments, GPU Kilosort4, Phy 2, drivers. |
| `HARDWARE.md` | The rig: devices, wiring, ADC channel map, sync rationale. |

---

## Make the sounds, then choose them

The tabs either emulates basic functions provided by Sound Analysis Pro to record audio, or synthesizes generated calls.
Then from a library of sounds, you can design an experiment with them.

**Tab 1 — Record Vox.** One input channel per bird. A file is saved whenever a bird calls,
starting two seconds *before* the sound (a ring buffer, so onsets are never
clipped) and closing after a second of silence. Every file is then measured and kept
only if it is really a vocalization: enough calls, of plausible duration, forming a long
enough bout, and not mostly low-frequency cage noise. **Rejected files are moved to a
`rejected/` folder with the reason recorded, never deleted** — so if the settings were
too strict you can loosen them and get the audio back. Recordings are filed by bird and
by day:

```
song_data/A2N3/2026-09-02/A2N3_20260902_094413_218_ch1.wav
                          rejected/...          (with the reason in events.csv)
                          events.csv            (every decision, kept or not)
```

**Tab 2 — Generate Calls.** Synthesizes begging calls with a conditional variational autoencoder (VAE) 
trained on 32,647 real calls from 9 chicks aged 4–15 days. Move the sliders, press **Listen** to hear the call and see its
sonogram, and **Save as stimulus** to put it in your sound folder — from there it is an
ordinary WAV and every later tab treats it exactly like a recording.

Two things make it especially useful:

- **Every preview is measured** with the same code as tab 2, and the numbers are shown
  under the sonogram. So you can compare a generated call against your real recordings
  directly instead of trusting the slider labels.
- **Sliders deliberately extend past what real chicks do.** Values outside the training
  range are marked *supernormal* in the interface and written into
  `stimulus_provenance.csv`, along with every parameter and the seed — a generated
  stimulus is as reproducible as a recorded one is traceable.

**Make a graded set** generates one stimulus per step of a chosen setting with everything
else fixed (including the seed). That is how a supernormal series is built.

**Tab 3 — Sound Library.** Browse everything you have recorded or generated, see each one's sonogram with its calls
marked, and plot any of the seven features under it. Tune what counts as a call with the
threshold and duration settings. Then drag across the sonogram, press *Play* to check it,
and *Use as stimulus* — the clip is faded, normalized, resampled to match your sound
folder, and written there, where **tab 3 picks it up immediately**. Every stimulus is
logged in `stimulus_provenance.csv` with the recording and offsets it came from.

### What it measures

| Feature | What it means |
|---------|---------------|
| Amplitude | loudness, in uncalibrated dB |
| Mean frequency | centre of the sound's energy |
| Pitch | harmonic pitch where the sound is harmonic, mean frequency otherwise |
| Goodness of pitch | how strongly harmonic the sound is |
| Wiener entropy | noisy (near 0) vs tonal (very negative); unaffected by loudness |
| FM | how fast the frequency is sweeping, in degrees |
| AM | how fast the loudness is changing; positive at onsets, negative at offsets |

### Tuned for begging calls

Defaults are set for **zebra finch chick begging calls**, not adult song. Measured from
the four sample stimuli, these calls carry energy to about 14 kHz, so the analysis band
runs to 15 kHz — a song-tuned 8 kHz ceiling would clip them. Their peak frequency also
moves from about 2 kHz to 7 kHz between day 7 and day 19, so **expect to save different
settings for different ages** rather than one set for everything.

### Hearing a sound before you use it

Anything FEATHER can see, it can play. On **tab 4** double-click a sound in the list to hear
it, or use *Play all in turn* to run through the whole folder back to back before committing
to a play order. Tab 3 plays either a dragged selection or the whole recording, and tab 2
plays each generated call as you make it, with *Play again* to repeat it. One **Stop** silences
whatever is sounding.

Auditioning always goes through the computer's speakers, never the Harp card — it is for
choosing stimuli, so avoid it while a recording is running.

### Playing to birds without ephys

Tab 5 asks where the sound comes out. The **Harp Sound Card** is the ephys path: it puts a
sample-accurate marker on ADC1, which is what every timing claim in this package rests on.
**This computer's speakers** is for behavior-only sessions before the ephys stage — same
play order, same logs, same saved experiment file, but onset times are software times
(good to a few milliseconds) with no marker on an acquisition clock. Use the card whenever
you are recording neurons.

---

## Quick start

**1. Install (once).** See **INSTALL.md**. In short:

```bat
:: main environment (playback control + analysis + Kilosort4)
conda env create -f environment-feather.yml
:: GPU spike sorting (CUDA build of PyTorch)
conda activate feather
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
:: manual curation (separate environment)
conda env create -f environment-phy2.yml
```

**2. Try it with no hardware.** Open the app:

```bat
conda activate feather
python playback_gui.py
```

On tab **4. Design Experiment** click *Build play order*, then on tab
**6. Preliminary Analysis** click *Generate demo data* → *Run spike sorting* →
*Show response (PSTH)*. This exercises the entire analysis path (onset detection →
Kilosort4 → PSTH) on a simulated recording. Prefer code? Do the same in
`notebooks/FEATHER_notebook.ipynb`.

**3. Run a real experiment.** Wire the rig (**HARDWARE.md**), put your WAVs in
`stimuli/` (or point the app at any folder), design the play order, load the
sounds onto the card, start Open Ephys recording, then press **Start** in FEATHER.
After recording: *Find sound onsets* → *Run spike sorting* → *Open in Phy* or
*Open in Jupyter*.

---

## The three environments

| Environment | Used for | Why separate |
|-------------|----------|--------------|
| **feather** (Python 3.11) | Everything in this package: design, playback control, onset detection, and Kilosort4 GPU spike sorting. | Kilosort4 needs a specific NumPy/PyTorch combo that must not be disturbed. |
| **phy2** (Python 3.11) | Phy 2 manual curation only. | Phy pins older Qt/OpenGL versions that conflict with Kilosort4's stack. |
| *(your recorder)* | Open Ephys GUI, running the acquisition board. | Separate application entirely; FEATHER reads its saved recordings. |

The app finds Phy automatically (it looks for a `phy2` conda env, or `phy` on your
PATH, or the `FEATHER_PHY` environment variable).

---

## Requirements at a glance

- **Windows** with an **NVIDIA GPU** (CUDA 12.1-capable) for Kilosort4.
- A **microphone and sound interface** for recording birds (one input channel per bird).
  Sound Bench uses PortAudio via `sounddevice`; on Windows, more than two channels
  depends on your interface and may need its ASIO or WASAPI driver.
- **Open Ephys GUI** with an acquisition board + I/O (ADC) board.
- A **Harp Sound Card** and its manufacturer TCP interface (`server_v1.0.exe`) —
  used only for *uploading* sounds; not redistributed here (see INSTALL.md).
- Miniconda / Anaconda.

Full details, verification steps, and troubleshooting are in **INSTALL.md**.
