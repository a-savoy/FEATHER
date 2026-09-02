# FEATHER — Installation & Setup

This guide sets up everything: the two Python environments, GPU Kilosort4, Phy 2,
and the hardware-side pieces. Tested on Windows 11 with an NVIDIA RTX 4060 Ti.

---

## 0. Prerequisites

1. **Miniconda or Anaconda** — <https://docs.conda.io/en/latest/miniconda.html>
2. **NVIDIA GPU + driver** capable of **CUDA 12.1** (needed for Kilosort4 GPU
   sorting). Check with `nvidia-smi`. CPU-only will run but is very slow.
3. **Open Ephys GUI** — <https://open-ephys.org/gui> — to record from the
   acquisition board. FEATHER does not record; it reads Open Ephys sessions.
4. **Harp Sound Card** and its manufacturer software:
   - The card is driven for playback directly over its COM serial port (no extra
     software needed for *triggering*).
   - **Uploading** sounds into card memory needs the manufacturer's TCP interface
     (`server_v1.0.exe`, listens on `localhost:9999`). This binary is **not
     redistributed** with FEATHER — obtain the "Harp Sound Card Interface TCP"
     package from the vendor and note where `server_v1.0.exe` lives. FEATHER
     looks for it at `%USERPROFILE%\Documents\HarpSoundCard\Interface TCP\
     server_v1.0.exe`; override with the `FEATHER_HARP_SERVER` environment
     variable if yours is elsewhere.
5. **The call generator (tab 2)** needs PyTorch, which you already install for
   Kilosort4. It runs on the **CPU** in well under a second per call, so the CUDA build
   is not required for it — a CPU-only machine can generate stimuli even though it
   cannot spike-sort. The trained model ships in `callgen/`; nothing else to install.

6. **A microphone per bird**, on a sound interface with enough input channels, if you
   want to record with FEATHER (tab 1, Record Vox). Sound Bench uses PortAudio through the
   `sounddevice` package, which `requirements.txt` installs. On Windows, stereo works
   with any card, but **more than two channels depends on your interface** — you may
   need its ASIO or WASAPI driver. Check what FEATHER can see with:

   ```bat
   python -c "import song_recorder; print(*song_recorder.list_devices(), sep=chr(10))"
   ```

7. An **FTDI USB-serial driver** (the Harp card enumerates as a COM port; Windows
   usually installs this automatically).

---

## 1. The main environment: `feather`

This runs the app, the notebook, playback control, onset detection, and
**Kilosort4**.

### Option A — conda environment file (recommended)

```bat
cd <path-to-FEATHER>
conda env create -f environment-feather.yml
conda activate feather
```

### Option B — manual

```bat
conda create -n feather -y python=3.11
conda activate feather
pip install -r requirements.txt
```

### GPU PyTorch for Kilosort4 (do this in either case)

`requirements.txt` intentionally leaves out PyTorch so you get the correct CUDA
build. Install it explicitly:

```bat
conda activate feather
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```bat
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

You should see `CUDA available: True` and your GPU name.

---

## 2. The curation environment: `phy2`

Phy 2 pins older Qt/OpenGL packages, so it **must** be its own environment.

### Option A — conda environment file (recommended)

```bat
conda env create -f environment-phy2.yml
```

### Option B — manual (Phy's official recipe)

```bat
conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt qtconsole requests responses scikit-learn scipy traitlets
conda activate phy2
pip install phy==2.1.0rc1
```

Verify:

```bat
conda activate phy2
phy --version
```

FEATHER finds Phy automatically. If yours is in a non-standard place, set:

```bat
setx FEATHER_PHY "C:\path\to\phy2\Scripts\phy.exe"
```

---

## 3. Run FEATHER

**The app:**

```bat
conda activate feather
cd <path-to-FEATHER>
python playback_gui.py
```

or double-click **`run_feather.bat`** (edit the env name inside if you didn't call
it `feather`).

**The notebook:**

```bat
conda activate feather
cd <path-to-FEATHER>
jupyter lab notebooks/FEATHER_notebook.ipynb
```

---

## 4. Smoke test (no hardware needed)

This confirms the whole analysis chain works before you touch the rig.

**In the app:** tab **4. Design Experiment** → *Build play order*; tab
**6. Preliminary Analysis** → *Generate demo data* (writes a fake 128-channel
Open Ephys recording matching your design) → *Run spike sorting* (Kilosort4 on the
GPU) → *Show response (PSTH)*. You should get ~10 units, a few of which respond
after each sound.

**On the command line** (equivalent):

```bat
conda activate feather
python make_test_data.py
```

Also run the Sound Bench tests, which need no hardware and take a few seconds:

```bat
python test_song_features.py
python test_song_recorder.py
python test_song_library.py
python test_song_generate.py
```

All four should end with `N/N passed`. These check the acoustic features against
signals whose answers are known by construction, the recorder's keep/discard decisions,
that exported stimuli land at the right sample rate for the Design tab, and that the call
generator renders, is reproducible from its seed, and flags supernormal settings.

This writes a synthetic recording under `test_data\...` with a `GROUND_TRUTH.csv`.
Point the app's *Recording* field at that `recording1` folder and analyze it, or
open `notebooks/FEATHER_notebook.ipynb` and run the analysis section on it.

---

## 5. Verify the hardware link (when the rig is connected)

**Harp Sound Card reachable** (safe — identifies only, plays nothing):

```bat
conda activate feather
python harp_control.py
```

Expect `WhoAmI = 1280` (the Sound Card's device ID). If you get
`could not open port 'COM4'` / *Access is denied*, another program (or a second
FEATHER window) is holding the port — see **Troubleshooting**.

**Upload path** (needs the vendor server): the app's *Load sounds onto the card*
button, or the notebook's upload cell, will auto-start `server_v1.0.exe` and push
each stimulus to a card memory index.

---

## Troubleshooting

**`could not open port 'COM4': Access is denied` (PermissionError 13).**
Only one program can hold the card's COM port. Causes, in order:
1. A second FEATHER window is open — run **only one at a time**; close the others.
2. A previous window was force-closed and left the port locked — **unplug the Harp
   card's USB, wait 5 s, plug it back in** (this does not disturb an Open Ephys
   recording; the sound card is a separate USB device).
3. The Harp configuration GUI is open — close it.
If after a replug the port shows in Device Manager but won't open (device in an
"error"/disabled state), right-click it in Device Manager → **Enable device**.

**Kilosort4 says CUDA is not available.** You installed the CPU build of PyTorch.
Reinstall with the `--index-url https://download.pytorch.org/whl/cu121` line above.

**No input devices listed, or recording will not start.** Run the `list_devices` command
above. If your interface is missing, it is a driver matter outside FEATHER. If it is listed
but recording fails, the channel count or sample rate you asked for is probably not
supported by that device — try 1 channel at 44100 Hz first, then widen. On Windows, a
device already open in another program cannot be opened again.

**Everything is rejected / nothing is recorded.** The trigger threshold (tab 1) is
relative to full scale; watch the level meters while a bird calls and set the threshold
just under the level you see. Rejected files are in the `rejected/` folder next to the
recordings, and `events.csv` gives the reason each one failed — that tells you which
setting to loosen.

**The app can't find Phy / Jupyter.** Make sure the `phy2` env exists (Phy) and
that you launched the app from the activated `feather` env (Jupyter is taken from
the same env that runs the app). You can also set `FEATHER_PHY`.

**Upload fails: "Could not start/reach the Harp TCP server."** The vendor
`server_v1.0.exe` isn't where FEATHER expects it. Set `FEATHER_HARP_SERVER` to its
full path, or place it under `Documents\HarpSoundCard\Interface TCP\`.

**No neurons / bad geometry after sorting.** The probe channel map must match your
wiring. Use the right `probe_*.json`, or rebuild with `build_probes.py`. See
HARDWARE.md.

---

## Tested versions (what this package was validated against)

- Python 3.11, Windows 11.
- `feather` env: numpy 1.26.4, scipy 1.17.1, pandas 3.0.3, soundfile 0.13.1,
  pyserial 3.5, matplotlib 3.7.3, scikit-learn 1.8.0, numba 0.65.1,
  probeinterface 0.3.2, **kilosort 4.1.7**, **torch 2.5.1+cu121**,
  faiss-cpu 1.14.3, jupyterlab 4.5.7, nbformat 5.10.4, Pillow 11.3.0.
- `phy2` env: **phy 2.1.0rc1**, phylib 2.7.0, PyQt5 5.15.11, PyOpenGL,
  numpy 2.x, scipy 1.17.1.
