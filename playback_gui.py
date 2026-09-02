"""playback_gui.py — a simple window for running the playback experiment.

Double-click "Playback Experiment (App)" on the Desktop, or run this file, to open it.
No coding needed: fill in the fields, click the buttons top to bottom.

It reuses the same tested code as the notebook:
    playback_tools.py  (build the play order)
    harp_upload.py     (load sounds onto the card)
    harp_control.py    (play sounds, timing signal)
"""
import os
import sys

# The desktop shortcut runs this with pythonw.exe, which has no console, so
# sys.stdout/sys.stderr are None. Libraries that print (e.g. Kilosort) would then
# crash with "'NoneType' object has no attribute 'write'". Send those writes to a log file.
if sys.stdout is None or sys.stderr is None:
    _logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_log.txt"),
                 "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _logf
    sys.stderr = sys.stderr or _logf

import time
import json
import glob
import queue
import random
import threading

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import soundfile as sf
import playback_tools as pt
import harp_control as hc
import harp_upload as hu
import song_tabs
from song_library import read_audio as slib_read

STIMULI_DIR = os.path.join(HERE, "stimuli")
OUTPUT_DIR = os.path.join(HERE, "output")
CONFIG_DIR = os.path.join(HERE, "configs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CARD_SR = 96000
GATE_OUTPUT = 0   # digital output wired to ADC1 (fixed by the rig wiring; DO0)
MIC_CHANNEL = "ADC3"   # spare ADC for an optional microphone at the bird (set to None to disable)
# Kilosort sorting parameters the user can tweak under Advanced. Each entry:
# (key, default, type, short label, plain-language description). These defaults are used every
# time unless the user deliberately turns on Advanced editing (meant only for initial tuning).
KS_PARAMS = [
    ("Th_universal", 9.0, float, "Detection threshold, first pass",
     "How strong a spike must be to be detected using Kilosort's built-in templates. "
     "Higher = fewer but cleaner spikes; lower = more spikes but more noise."),
    ("Th_learned", 8.0, float, "Detection threshold, final pass",
     "The same idea for the second pass, using templates learned from your own data. "
     "Usually set a little lower than the first-pass threshold."),
    ("Th_single_ch", 6.0, float, "Single-channel threshold",
     "Threshold for spikes that appear mostly on a single channel. Lower values catch small, "
     "very local units."),
    ("nblocks", 1, int, "Drift-correction blocks",
     "How finely the probe is split to track the tissue drifting over time. 0 = off (rigid), "
     "1 = the whole probe as one block, more = finer. Use 0 for short, stable recordings."),
    ("batch_size", 60000, int, "Batch size (samples)",
     "How many samples are processed at once (60000 is about 2 seconds at 30 kHz). Lower this "
     "only if the graphics card runs out of memory."),
    ("highpass_cutoff", 300.0, float, "High-pass filter (Hz)",
     "Frequencies below this are filtered out before sorting, removing slow drift while keeping "
     "the spike band."),
    ("dminx", 32.0, float, "Horizontal channel spacing (um)",
     "The horizontal distance between electrode columns, used when grouping channels into "
     "templates. Match this to your probe's column spacing."),
    ("min_template_size", 10.0, float, "Smallest template size (um)",
     "The spatial size of the smallest spike template Kilosort will use. Smaller values fit "
     "tight, local spikes."),
    ("nearest_chans", 10, int, "Nearby channels per template",
     "How many neighboring channels are used to describe each unit. More channels give each "
     "unit a broader spatial footprint."),
    ("duplicate_spike_ms", 0.25, float, "Duplicate-spike window (ms)",
     "Two spikes from the same unit closer together than this are treated as one (a duplicate) "
     "and merged."),
]
KS_PARAM_DEFAULTS = {k: d for k, d, _c, _l, _ds in KS_PARAMS}
OE_ROOTS = [r"C:\Users\asavoy\Documents\Open Ephys",
            r"C:\Users\asavoy\Desktop\OpenEphys_Data"]


class App(song_tabs.SoundBenchTabs):
    def __init__(self, root):
        self.root = root
        root.title("F.E.A.T.H.E.R.")
        root.geometry("880x920")
        root.minsize(360, 260)
        try:
            root.iconbitmap(os.path.join(HERE, "feather.ico"))
        except Exception:
            pass
        self.stimuli_dir = STIMULI_DIR
        self.stim_table = None
        self.schedule = None
        self.file_index = {}
        self.card_waves = {}
        self.sr = None
        self.stop_event = threading.Event()
        self.q = queue.Queue()
        self.sortq = queue.Queue()
        self.log = []
        self.ks_dir = None
        self.an_onsets = None
        self.an_fs = 30000.0
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_start_epoch = time.time()
        self._sb_init()          # Sound Bench (Record + Sound Library tabs)
        self.session_log = dict(session_id=self.session_id,
                                session_started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                events=[])
        self._save_session_log()

        header = ttk.Frame(root)
        header.pack(fill="x", padx=10, pady=(8, 0))
        try:
            self._feather_photo = tk.PhotoImage(file=os.path.join(HERE, "feather_72.png"))
            ttk.Label(header, image=self._feather_photo).pack(side="left", padx=(0, 10))
        except Exception:
            self._feather_photo = None
        htxt = ttk.Frame(header)
        htxt.pack(side="left")
        ttk.Label(htxt, text="F.E.A.T.H.E.R.", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(htxt, text="Friendly Ephys Acquisition Terminal Helping Everyone Record",
                  foreground="gray").pack(anchor="w")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_record = ttk.Frame(nb)
        self.tab_library = ttk.Frame(nb)
        self.tab_generate = ttk.Frame(nb)
        self.tab_setup = ttk.Frame(nb)
        self.tab_run = ttk.Frame(nb)
        self.tab_analyze = ttk.Frame(nb)
        self.tab_saved = ttk.Frame(nb)
        self.tab_help = ttk.Frame(nb)
        self.tab_about = ttk.Frame(nb)
        nb.add(self.tab_about, text="0. About")
        nb.add(self.tab_record, text="1. Record Vox")
        nb.add(self.tab_generate, text="2. Generate Calls")
        nb.add(self.tab_library, text="3. Sound Library")
        nb.add(self.tab_setup, text="4. Design Experiment")
        nb.add(self.tab_run, text="5. Run Experiment")
        nb.add(self.tab_analyze, text="6. Preliminary Analysis")
        nb.add(self.tab_saved, text="7. Saved Files")
        nb.add(self.tab_help, text="8. To Jupyter and Beyond!")
        # Design Experiment is built first: the library tab exports into its sound
        # folder, so self.v_stimdir must already exist.
        self._build_setup(self._make_scrollable(self.tab_setup))
        self._build_record(self._make_scrollable(self.tab_record))
        self._build_generate(self._make_scrollable(self.tab_generate))
        self._build_library(self._make_scrollable(self.tab_library))
        self._build_run(self._make_scrollable(self.tab_run))
        self._build_analyze(self._make_scrollable(self.tab_analyze))
        self._build_saved(self._make_scrollable(self.tab_saved))
        self._build_help(self._make_scrollable(self.tab_help))
        self._build_about(self._make_scrollable(self.tab_about))
        self.refresh_sounds()

    def _help_section(self, parent, title, body):
        ttk.Label(parent, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        ttk.Label(parent, text=body, justify="left", wraplength=820).pack(anchor="w")

    # ---------- session log + saved-files tab ----------
    def _save_session_log(self):
        self.session_log["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(os.path.join(OUTPUT_DIR, f"session_log_{self.session_id}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(self.session_log, f, indent=2)
        except Exception:
            pass

    def _log_event(self, kind, **details):
        self.session_log["events"].append(
            dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), event=kind, **details))
        self._save_session_log()

    def _build_saved(self, tab):
        ttk.Label(tab, text="Saved files (this session)",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Label(tab, justify="left", wraplength=820, text=(
            "Everything the app has saved this session, and where to find it. The session log "
            "(session_log_...json) records everything you did - all your settings and the start/stop times "
            "of each step. Click Refresh after each step to update this list.")).pack(anchor="w")
        ttk.Button(tab, text="Refresh", command=self.refresh_saved).pack(anchor="w", pady=4)
        self.saved_box = tk.Text(tab, height=22, width=92, state="disabled")
        self.saved_box.pack(fill="both", expand=True, pady=2)
        self.refresh_saved()

    def _scan_saved_files(self):
        out = OUTPUT_DIR
        known = [
            ("Session log (everything you did)", os.path.join(out, f"session_log_{self.session_id}.json")),
            ("Play order", os.path.join(out, "schedule.csv")),
            ("Experiment metadata (latest run)", os.path.join(out, "experiment_info_latest.json")),
            ("Run log (what actually played)", os.path.join(out, "trigger_log.csv")),
            ("Sound onsets (latest)", os.path.join(out, "sound_onsets_latest.csv")),
        ]
        for p in sorted(glob.glob(os.path.join(out, "experiments", "experiment_*.json"))):
            known.append(("Experiment metadata (per run)", p))
        for p in sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json"))):
            known.append(("Saved settings", p))
        rec = self._resolve_recording()
        if rec:
            known += [
                ("Sound onsets (saved with the recording)", os.path.join(rec, "sound_onsets.csv")),
                ("Sorted spikes (Kilosort output)", os.path.join(rec, "kilosort4")),
                ("Analysis notebook", os.path.join(rec, "analysis.ipynb")),
            ]
        return [(d, p) for d, p in known if os.path.exists(p)]

    def refresh_saved(self):
        items = self._scan_saved_files()
        lines = [f"Session started: {self.session_log['session_started']}",
                 f"Output folder:   {OUTPUT_DIR}", ""]
        if not items:
            lines.append("Nothing saved yet. Build a play order, run the experiment, or analyze a recording.")
        for desc, p in items:
            try:
                m = os.path.getmtime(p)
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(m))
                tag = "   <- this session" if m >= self.session_start_epoch else ""
            except OSError:
                when, tag = "", ""
            lines.append(f"{desc}:")
            lines.append(f"    {p}    ({when}){tag}")
        self.say(self.saved_box, "\n".join(lines))

    def _build_help(self, tab):
        ttk.Label(tab, text="To Jupyter and beyond!", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Label(tab, justify="left", wraplength=820, text=(
            "Once you have sorted your spikes on the Analyze tab, you continue your work in two other "
            "programs. Both are one click away, on the Analyze tab under Step 4.")).pack(anchor="w")

        self._help_section(tab, "Curate your units in Phy", (
            "Click 'Open in Phy' (Analyze tab, Step 4). Phy is a separate program for checking the automatic "
            "sorting by eye: look at each unit's waveform and firing, merge units that are really the same "
            "neuron, split ones that got combined, and label units as good, MUA (multi-unit), or noise. Your "
            "changes save back into the recording's 'kilosort4' folder, and this app and the notebook will "
            "then use your curated labels. Close Phy when you're done."))

        self._help_section(tab, "Do your own analysis in Jupyter", (
            "Click 'Open in Jupyter' (Analyze tab, Step 4). This opens a notebook ('analysis.ipynb', saved "
            "next to your recording) already filled in with the paths to your spikes and sounds, plus example "
            "code you can build on: a PSTH function, a way to compare responses to different sounds, and each "
            "unit's firing rate. Run the cells top to bottom, then add your own analysis - response latencies, "
            "tuning across sounds, spike-waveform plots, and so on."))

        self._help_section(tab, "For the full details", (
            "See the 'About' tab for a complete explanation of how the app works, how the sound timing is "
            "kept exact, and where everything is saved."))

    def _build_about(self, tab):
        try:
            self._about_photo = tk.PhotoImage(file=os.path.join(HERE, "feather_256.png"))
            ttk.Label(tab, image=self._about_photo).pack(anchor="w", pady=(6, 2))
        except Exception:
            self._about_photo = None
        ttk.Label(tab, text="F.E.A.T.H.E.R.", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(tab, text="Friendly Ephys Acquisition Terminal Helping Everyone Record",
                  font=("Segoe UI", 11, "italic")).pack(anchor="w", pady=(0, 8))
        ttk.Label(tab, justify="left", wraplength=820, text=(
            "This app runs a sound-playback experiment during an Open Ephys neural recording and then "
            "analyzes it - from playing the sounds all the way to sorted neurons and their responses.")).pack(anchor="w")

        self._help_section(tab, "How the timing stays exact", (
            "When a sound plays, the sound card sends a marker on a digital output that is recorded on the "
            "ADC1 input, and a copy of the sound itself is recorded on ADC2. Because these are recorded on "
            "the same clock as the neural data, the exact time of every sound is stored inside the recording. "
            "That is what lets the app line up sounds with neural activity to about one sample (0.03 ms). The "
            "app rechecks this timing on every recording and warns you if it drifts."))

        self._help_section(tab, "Tab 1 - Record Vox", (
            "Records the birds. Choose the microphone and how many channels (one bird per channel), name "
            "the birds, and press 'Start recording'. A file is saved whenever a bird calls - starting a "
            "couple of seconds BEFORE the sound, so onsets are never clipped - and stops after a second of "
            "silence. Each file is then measured to check it really is a vocalization rather than cage "
            "noise; anything rejected is moved to a 'rejected' folder with the reason, never deleted, so "
            "you can loosen the settings and get it back. Recordings are filed by bird and by day."))

        self._help_section(tab, "Tab 2 - Generate Calls", (
            "Makes new begging calls with a model trained on 32,647 real calls from chicks aged 4-15 "
            "days. Move the sliders and press Listen: you hear the call and see its sonogram, plus the "
            "same measurements the Sound Library tab reports, so you can compare a generated call "
            "against your real recordings directly. 'Save as stimulus' puts it in your sound folder "
            "like any recorded clip. Sliders deliberately go beyond what real chicks do - those values "
            "are marked 'supernormal' and recorded in the stimulus log. 'Make a graded set' generates "
            "one stimulus per step of a chosen setting, with everything else including the seed held "
            "fixed, which is how a supernormal series is built. Note the brightness slider grades "
            "evenly; the pitch slider changes the sound but not in even steps."))

        self._help_section(tab, "Tab 3 - Sound Library", (
            "Look through your recordings and turn the good ones into stimuli. 'Find recordings' lists what "
            "is in the folder; click one to see its sonogram with the calls marked underneath, and pick "
            "which measurement to plot (amplitude, pitch, entropy, and so on). 'What counts as a call' "
            "controls the splitting - raise the threshold to keep only the loudest calls. Drag across the "
            "sonogram to choose a stretch, press Play to hear it, then 'Use as stimulus' to save it into "
            "your sound folder, where the Design Experiment tab will pick it up. 'Measure all' builds a "
            "table of every call in the folder, which you can save as a CSV."))

        self._help_section(tab, "Tab 4 - Design experiment", (
            "Choose the folder with your sound files. Double-click any sound to hear it, or select "
            "it and press Play; 'Play all in turn' plays the whole folder back to back so you can "
            "compare them before committing to a play order. Auditioning uses the computer's "
            "speakers - the experiment itself plays through the Harp card. Then set the play order: how many times each sound "
            "plays, whether the order is shuffled, and the gaps between sounds. You must also choose whether "
            "every run of the experiment is Identical (same order and gaps every time) or Different (a new "
            "random order each run). Click 'Build play order' to create the list (saved as schedule.csv), "
            "then 'Load sounds onto the card'. Each run's choice, random seed, settings, and exact order are "
            "saved in output/experiments/experiment_<date_time>.json. You can also save and reload settings."))

        self._help_section(tab, "Tab 5 - Run experiment", (
            "Choose where the sound comes out first: the Harp Sound Card (for ephys - it puts a "
            "sample-accurate marker on ADC1) or this computer's speakers (for behaviour-only sessions "
            "before the ephys stage, where there is no marker and onset times are software times, good "
            "to a few milliseconds). "
            "Press Record in Open Ephys, then click 'Start experiment'. The sounds play in the order you "
            "built, and a progress bar shows how far along you are. Each sound's exact time is written into "
            "the recording. Press Stop to end early - it silences the speaker and saves what already played."))

        self._help_section(tab, "Tab 6 - Preliminary Analysis", (
            "Choose a recording (or leave blank for the most recent). 'Find sound onsets' locates every sound "
            "and labels it - read from the recorded audio, so it is correct even if the schedule changed - and "
            "checks the timing. 'Run spike sorting' runs Kilosort4 on your graphics card to find neurons; it "
            "uses fixed default sorting parameters (change them only for initial tuning, under Advanced - keep "
            "them the same for a real analysis). No real data yet? 'Generate demo data' makes a fake recording "
            "that matches your experiment so you can try the whole analysis first. 'Show response (PSTH)' plots "
            "one neuron's firing around the sounds; 'List units' shows them all and which respond."))

        self._help_section(tab, "Where your results are saved", (
            "- sound_onsets.csv (in the recording folder, plus a copy in the app's 'output' folder): each "
            "sound's exact start time, end time, and identity.\n"
            "- the 'kilosort4' folder (inside the recording folder): the sorted spikes (also what Phy reads).\n"
            "- analysis.ipynb (in the recording folder): your custom-analysis notebook.\n"
            "- schedule.csv and trigger_log.csv (in 'output'): the planned and actual play order."))

        self._help_section(tab, "Good to know", (
            "- Spike sorting and PSTH plots use your graphics card.\n"
            "- Keep using the analog amplifier and speaker - that is what keeps the sound timing exact. A "
            "Bluetooth or powered speaker would add a variable delay.\n"
            "- If you wire a microphone at the bird into a spare ADC channel, the Analyze step will also "
            "report the true sound-at-bird delay.\n"
            "- To practice without real data, run make_test_data.py, then in Analyze browse to the folder it "
            "prints and run steps 1 to 4."))

    def _make_scrollable(self, parent):
        """Wrap a tab in a scrolling area (vertical + horizontal scroll bars and
        mouse wheel) so the window can be resized to any size without hiding content."""
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner)

        def _on_canvas(event):
            # fill the width when the window is wide; keep the content's own width
            # (so the horizontal scroll bar works) when the window is narrow
            canvas.itemconfigure(window, width=max(inner.winfo_reqwidth(), event.width))
        canvas.bind("<Configure>", _on_canvas)

        def _wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    # ---------- helpers ----------
    def _grid_row(self, parent, r, label, var, width=10, hint=""):
        # label sizes to its full text (column 0), so it is never cut off;
        # entry in column 1; a gray formatting hint in column 2.
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=r, column=1, sticky="w", pady=3)
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(row=r, column=2, sticky="w", padx=(8, 0))

    def say(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    # ---------- Tab 1: setup ----------
    def _build_setup(self, tab):
        top = ttk.Frame(tab)
        top.pack(fill="x", pady=(6, 2))
        ttk.Label(top, text="Sound folder:").pack(side="left")
        self.v_stimdir = tk.StringVar(value=self.stimuli_dir)
        ttk.Entry(top, textvariable=self.v_stimdir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Browse...", command=self.browse_stimdir).pack(side="left")

        ttk.Label(tab, text="Sound files found in this folder:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 2))
        ttk.Label(tab, foreground="gray", wraplength=800, justify="left",
                  text="Double-click a sound to hear it, or select it and press Play. This "
                       "listens through the computer's speakers so you can choose before the "
                       "experiment; the experiment itself plays through the Harp card."
                  ).pack(anchor="w", pady=(0, 4))

        self.stim_tree = ttk.Treeview(tab, columns=("file", "seconds"), show="headings",
                                      height=7)
        self.stim_tree.heading("file", text="Sound")
        self.stim_tree.heading("seconds", text="Seconds")
        self.stim_tree.column("file", width=520, anchor="w")
        self.stim_tree.column("seconds", width=80, anchor="e")
        self.stim_tree.pack(fill="x")
        self.stim_tree.bind("<Double-1>", lambda e: self.play_selected_stimulus())

        prow = ttk.Frame(tab); prow.pack(fill="x", pady=3)
        ttk.Button(prow, text="Play", command=self.play_selected_stimulus).pack(side="left")
        ttk.Button(prow, text="Stop", command=self._audio_stop).pack(side="left", padx=4)
        ttk.Button(prow, text="Play all in turn",
                   command=self.play_all_stimuli).pack(side="left", padx=(0, 4))
        ttk.Button(prow, text="Refresh list", command=self.refresh_sounds).pack(side="left")
        self.stim_play_lbl = ttk.Label(prow, text="", foreground="gray")
        self.stim_play_lbl.pack(side="left", padx=10)

        self.sound_box = tk.Text(tab, height=3, width=76, state="disabled")
        self.sound_box.pack(fill="x", pady=(2, 0))

        ttk.Separator(tab).pack(fill="x", pady=6)
        ttk.Label(tab, text="Play-order settings",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.v_repeats = tk.StringVar(value="5")
        self.v_gapset = tk.StringVar(value="5, 10, 15, 20, 25")
        self.v_leadin = tk.StringVar(value="5")
        self.v_tail = tk.StringVar(value="5")
        self.v_seed = tk.StringVar(value="42")
        self.v_shuffle = tk.BooleanVar(value=True)
        self.v_norepeat = tk.BooleanVar(value=False)

        grid = ttk.Frame(tab)
        grid.pack(fill="x", pady=2)
        self._grid_row(grid, 0, "Times each sound plays:", self.v_repeats, hint="a whole number")
        self._grid_row(grid, 1, "Gaps between sounds (seconds):", self.v_gapset, width=24,
                       hint="separate with commas, e.g. 5, 10, 15")
        self._grid_row(grid, 2, "Silence at start (seconds):", self.v_leadin, hint="seconds (decimals ok)")
        self._grid_row(grid, 3, "Silence at end (seconds):", self.v_tail, hint="seconds (decimals ok)")
        self._grid_row(grid, 4, "Random seed:", self.v_seed,
                       hint="any whole number; same number = same order")
        ttk.Checkbutton(tab, text="Shuffle the order", variable=self.v_shuffle).pack(anchor="w")
        ttk.Checkbutton(tab, text="Never play the same sound twice in a row",
                        variable=self.v_norepeat).pack(anchor="w")

        ttk.Label(tab, text="Repeats of this experiment (you must choose one):",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        self.v_repeat_mode = tk.StringVar(value="")
        ttk.Radiobutton(tab, variable=self.v_repeat_mode, value="identical",
                        text="Identical every run - same order and gaps every time (uses the seed above)"
                        ).pack(anchor="w")
        ttk.Radiobutton(tab, variable=self.v_repeat_mode, value="different",
                        text="Different every run - a new random order and seed each time").pack(anchor="w")

        cfgf = ttk.Frame(tab)
        cfgf.pack(fill="x", pady=(6, 0))
        ttk.Label(cfgf, text="Saved settings:").pack(side="left")
        ttk.Button(cfgf, text="Save settings...", command=self.save_config).pack(side="left", padx=4)
        ttk.Button(cfgf, text="Load settings...", command=self.load_config).pack(side="left")

        ttk.Button(tab, text="Build play order", command=self.build_order).pack(anchor="w", pady=(8, 2))
        self.order_box = tk.Text(tab, height=4, width=76, state="disabled")
        self.order_box.pack(fill="x")

        ttk.Separator(tab).pack(fill="x", pady=6)
        ttk.Button(tab, text="Load sounds onto the card", command=self.load_sounds).pack(anchor="w")
        self.load_box = tk.Text(tab, height=5, width=76, state="disabled")
        self.load_box.pack(fill="x", pady=2)

    def browse_stimdir(self):
        # A file picker (not a folder picker) so the user can SEE the WAV files. We use the
        # chosen file's folder as the sound folder.
        f = filedialog.askopenfilename(
            title="Go to your sound folder and pick any WAV file in it",
            initialdir=(self.v_stimdir.get() or HERE),
            filetypes=[("Sound files", "*.wav"), ("All files", "*.*")])
        if f:
            self.v_stimdir.set(os.path.dirname(f))
            self.refresh_sounds()

    def refresh_sounds(self):
        self.stimuli_dir = self.v_stimdir.get().strip() or STIMULI_DIR
        if hasattr(self, "stim_tree"):
            self.stim_tree.delete(*self.stim_tree.get_children())
        try:
            self.stim_table = pt.scan_stimuli(self.stimuli_dir)
            self.sr = pt.check_uniform_sample_rate(self.stim_table)
            if hasattr(self, "stim_tree"):
                for _, r in self.stim_table.iterrows():
                    self.stim_tree.insert("", "end",
                                          values=(r["file"], f"{r['duration_s']:.2f}"))
            self.say(self.sound_box, f"{len(self.stim_table)} sound(s), all at {self.sr} Hz.")
        except Exception as e:
            self.stim_table = None
            self.say(self.sound_box, f"No usable sounds found in:\n{self.stimuli_dir}\n\n{e}")

    def _selected_stimulus_path(self):
        sel = self.stim_tree.selection() if hasattr(self, "stim_tree") else None
        if not sel:
            return None
        name = self.stim_tree.item(sel[0], "values")[0]
        return os.path.join(self.stimuli_dir, name)

    def play_selected_stimulus(self):
        path = self._selected_stimulus_path()
        if not path:
            messagebox.showinfo("Nothing chosen", "Click a sound in the list first.")
            return
        self._audio_play_file(path)

    def play_all_stimuli(self):
        """Audition the whole folder back to back, so you can compare them in one pass."""
        if self.stim_table is None or not len(self.stim_table):
            messagebox.showinfo("No sounds", "There are no sounds in this folder yet.")
            return
        try:
            import numpy as _np
            gap = None
            chunks = []
            for _, r in self.stim_table.iterrows():
                y, fs = slib_read(os.path.join(self.stimuli_dir, r["file"]))
                if gap is None:
                    gap = _np.zeros(int(0.4 * fs))
                    sr = fs
                chunks += [y, gap]
            self._audio_play(_np.concatenate(chunks), sr,
                             f"all {len(self.stim_table)} sounds in turn")
        except Exception as e:
            messagebox.showwarning("Could not play them", str(e))

    def _make_schedule(self, seed):
        gaps = [float(x) for x in self.v_gapset.get().replace(",", " ").split()]
        return pt.build_schedule(
            self.stim_table, repeats=int(self.v_repeats.get()),
            randomize_order=self.v_shuffle.get(), avoid_immediate_repeat=self.v_norepeat.get(),
            gap_mode="set", gap_set=gaps,
            lead_in=float(self.v_leadin.get()), tail=float(self.v_tail.get()), seed=seed)

    def build_order(self):
        if self.stim_table is None:
            messagebox.showwarning("No sounds", "No sound files found. Click 'Refresh list' first.")
            return
        mode = self.v_repeat_mode.get()
        if mode == "":
            messagebox.showwarning("Choose one first",
                                   "Under 'Repeats of this experiment', choose whether every run should be:\n\n"
                                   "  - Identical every run (same order and gaps), or\n"
                                   "  - Different every run (a new random order each time).")
            return
        try:
            if mode == "different":
                seed = random.randint(0, 999999)
                self.v_seed.set(str(seed))
            else:
                seed = int(self.v_seed.get())
            self.schedule = self._make_schedule(seed)
            self.schedule.to_csv(os.path.join(OUTPUT_DIR, "schedule.csv"), index=False)
            self.file_index = {f: 2 + i for i, f in enumerate(self.stim_table["file"])}
            total = float(self.schedule["offset_s"].iloc[-1] + self.schedule["gap_after_s"].iloc[-1])
            mm, ss = divmod(int(round(total)), 60)
            note = ("A NEW random order will be made for each run." if mode == "different"
                    else f"Every run will be IDENTICAL (seed {seed}).")
            self.say(self.order_box,
                     f"Play order ready: {len(self.schedule)} sounds, total time {mm}:{ss:02d} (min:sec).\n"
                     f"{note}  Saved to output/schedule.csv.")
            self._log_event("play_order_built", repeat_mode=mode, seed=int(seed),
                            n_sounds=int(len(self.schedule)),
                            settings=dict(repeats=self.v_repeats.get(), gaps=self.v_gapset.get(),
                                          lead_in=self.v_leadin.get(), tail=self.v_tail.get(),
                                          shuffle=bool(self.v_shuffle.get()),
                                          no_repeat=bool(self.v_norepeat.get()),
                                          sound_folder=self.stimuli_dir))
        except Exception as e:
            messagebox.showerror("Could not build play order", str(e))

    def _config_vars(self):
        """All the GUI settings, as a name -> variable map (used to save/load)."""
        return {
            "sound_folder": self.v_stimdir,
            "repeats": self.v_repeats, "gaps": self.v_gapset,
            "lead_in": self.v_leadin, "tail": self.v_tail, "seed": self.v_seed,
            "shuffle": self.v_shuffle, "no_repeat": self.v_norepeat,
            "repeat_mode": self.v_repeat_mode,
            **{"ks_" + k: var for k, var in self.ks_params.items()},
        }

    def save_config(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save settings as", initialdir=CONFIG_DIR,
            defaultextension=".json", filetypes=[("Settings file", "*.json")])
        if not path:
            return
        cfg = {k: v.get() for k, v in self._config_vars().items()}
        try:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            messagebox.showinfo("Saved", f"Settings saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Could not save settings", str(e))

    def load_config(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load settings", initialdir=CONFIG_DIR,
            filetypes=[("Settings file", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Could not load settings", str(e))
            return
        for k, var in self._config_vars().items():
            if k in cfg:
                var.set(cfg[k])
        self.refresh_sounds()      # reload the sound list for the loaded folder
        messagebox.showinfo("Loaded",
                            f"Loaded settings from:\n{os.path.basename(path)}\n\n"
                            'Now click "Build play order" to apply them.')

    def load_sounds(self):
        if self.schedule is None:
            messagebox.showwarning("Build first", "Click 'Build play order' before loading sounds.")
            return
        self.say(self.load_box, "Preparing and loading sounds... please wait.")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            files = list(self.stim_table["file"])
            for f in files:
                x, sr = sf.read(os.path.join(self.stimuli_dir, f), dtype="float32", always_2d=True)
                self.card_waves[f] = hu.wav_to_card_wave_int(x[:, 0], sr, CARD_SR)
            if not hu.ensure_server():
                self.q.put(("load", "Could not start the helper program. Check server_v1.0.exe."))
                return
            msg = ["Helper program ready.", ""]
            for f in files:
                ok = hu.upload_sound(self.card_waves[f], self.file_index[f], sample_rate=CARD_SR)
                msg.append(("loaded  " if ok else "FAILED  ") + f"slot {self.file_index[f]}: {f}")
            msg.append("\nDone. Sounds are on the card.")
            self._log_event("sounds_loaded", n_sounds=len(files), sound_folder=self.stimuli_dir)
            self.q.put(("load", "\n".join(msg)))
        except Exception as e:
            self.q.put(("load", f"Problem loading sounds:\n{e}"))
        self.root.after(50, self._drain_load)

    def _drain_load(self):
        try:
            while True:
                kind, text = self.q.get_nowait()
                if kind == "load":
                    self.say(self.load_box, text)
        except queue.Empty:
            pass

    # ---------- Tab 4: run ----------
    def _build_run(self, tab):
        out = ttk.LabelFrame(tab, text="Where the sound comes out")
        out.pack(fill="x", pady=(6, 6))
        self.v_playback = tk.StringVar(value="card")
        ttk.Radiobutton(out, variable=self.v_playback, value="card",
                        text="Harp Sound Card - for recording with Open Ephys (sample-accurate marker "
                             "on ADC1)").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Radiobutton(out, variable=self.v_playback, value="speakers",
                        text="This computer's speakers - for behaviour-only sessions, no ephys "
                             "(no timing marker)").pack(anchor="w", padx=8, pady=(0, 6))
        ttk.Label(out, foreground="gray", wraplength=800, justify="left",
                  text="The play order, logging and saved experiment file are identical either way. "
                       "Only the Harp card records a marker on the ephys clock, so use it whenever "
                       "you are recording neurons.").pack(anchor="w", padx=8, pady=(0, 6))

        ttk.Label(tab, text="Before you start:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 2))
        ttk.Label(tab, justify="left", text=(
            "1. In Open Ephys, make sure ADC1 and ADC2 are on, then press Record.\n"
            "   (Skip this if you chose the computer's speakers.)\n"
            "2. Click Start. This plays sound out loud.\n"
            "3. When it finishes, stop the recording in Open Ephys.")).pack(anchor="w", pady=(0, 6))

        self.start_btn = ttk.Button(tab, text="Start experiment", command=self.start_run)
        self.start_btn.pack(anchor="w", pady=(4, 2))
        self.stop_btn = ttk.Button(tab, text="Stop", command=self.stop_run, state="disabled")
        self.stop_btn.pack(anchor="w")

        self.progress = ttk.Progressbar(tab, length=560, mode="determinate", maximum=100)
        self.progress.pack(anchor="w", pady=(10, 2))
        self.status_lbl = ttk.Label(tab, text="Not started.", font=("Segoe UI", 10))
        self.status_lbl.pack(anchor="w")

    def start_run(self):
        if self.schedule is None:
            messagebox.showwarning("Build first", "Build the play order (tab 4) before starting.")
            return
        mode = self.v_repeat_mode.get()
        if mode == "":
            messagebox.showwarning("Choose one first",
                                   "On tab 4, choose whether runs are Identical or Different, then Build "
                                   "play order.")
            return
        if not messagebox.askyesno("Play sound?",
                                   "This will play the sounds out loud through the speaker.\n\nStart now?"):
            return
        if mode == "different":
            # a fresh random order for THIS run, so every run differs
            seed = random.randint(0, 999999)
            self.v_seed.set(str(seed))
            self.schedule = self._make_schedule(seed)
            self.schedule.to_csv(os.path.join(OUTPUT_DIR, "schedule.csv"), index=False)
        else:
            seed = int(self.v_seed.get())
        self._write_experiment_metafile(mode, seed)
        self._log_event("run_started", repeat_mode=mode, seed=int(seed),
                        n_sounds=int(len(self.schedule)), playback=self.v_playback.get())
        self.stop_event.clear()
        self.log = []
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        if self.v_playback.get() == "speakers":
            threading.Thread(target=self._run_worker_speakers, daemon=True).start()
        else:
            threading.Thread(target=self._run_worker, args=(GATE_OUTPUT,), daemon=True).start()
        self.root.after(100, self._poll_run)

    def _write_experiment_metafile(self, mode, seed):
        """Record this experiment's reproducibility choice, seed, settings, and play order.
        Saved timestamped under output/experiments/ (kept per run) plus a 'latest' copy."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        info = dict(
            repeat_mode=mode,
            repeat_mode_meaning=("same order and gaps every run" if mode == "identical"
                                 else "a new random order each run"),
            seed=int(seed),
            timestamp=ts,
            sound_folder=self.stimuli_dir,
            settings=dict(repeats=self.v_repeats.get(), gaps=self.v_gapset.get(),
                          lead_in=self.v_leadin.get(), tail=self.v_tail.get(),
                          shuffle=bool(self.v_shuffle.get()), no_repeat=bool(self.v_norepeat.get())),
            n_sounds=int(len(self.schedule)),
            order=[dict(trial=int(r.trial), file=str(r.file), gap_after_s=float(r.gap_after_s))
                   for r in self.schedule.itertuples()],
        )
        d = os.path.join(OUTPUT_DIR, "experiments")
        os.makedirs(d, exist_ok=True)
        for path in (os.path.join(d, f"experiment_{ts}.json"),
                     os.path.join(OUTPUT_DIR, "experiment_info_latest.json")):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

    def stop_run(self):
        self.stop_event.set()

    def _run_worker(self, gate_do):
        try:
            sched = self.schedule
            onsets = sched["onset_s"].tolist()
            total_s = float(sched["offset_s"].iloc[-1] + sched["gap_after_s"].iloc[-1])
            N = len(sched)
            with hc.SoundCard() as sc:
                if sc.whoami()[0] != 1280:
                    self.q.put(("run_error", "Sound card not responding on COM4.")); return
                sc.configure_gate(gate_do, hc.DO_HIGH_WHEN_SOUND)
                t0 = time.perf_counter()
                i = 0
                cur = ""
                while True:
                    if self.stop_event.is_set():
                        sc.stop()
                        self.q.put(("run_stop", (i, N))); break
                    elapsed = time.perf_counter() - t0
                    while i < N and elapsed >= onsets[i]:
                        r = sched.iloc[i]
                        idx = self.file_index[r["file"]]
                        sc.play(idx)
                        cur = r["file"]
                        self.log.append(dict(trial=int(r["trial"]), file=r["file"],
                                             sound_index=idx, onset_s=round(elapsed, 4)))
                        i += 1
                    if i >= N and elapsed >= total_s:
                        self.q.put(("run_done", (N, total_s))); break
                    self.q.put(("run_progress", (elapsed, total_s, i, N, cur)))
                    time.sleep(0.1)
            pd.DataFrame(self.log).to_csv(os.path.join(OUTPUT_DIR, "trigger_log.csv"), index=False)
        except Exception as e:
            self.q.put(("run_error", str(e)))

    def _run_worker_speakers(self):
        """Play the schedule through the computer's sound output.

        For behavioural sessions before the ephys stage. Everything else matches the
        Harp path - same schedule, same trigger_log.csv - except that there is no
        marker on the acquisition clock, so onset times here are software times and
        are only as good as the audio device's latency (a few ms), not sample-accurate.
        """
        try:
            import sounddevice as sd
            sched = self.schedule
            waves = {}
            sr = None
            for f in sched["file"].unique():
                x, fs = sf.read(os.path.join(self.stimuli_dir, f), dtype="float32", always_2d=True)
                waves[f] = x[:, 0]
                if sr is None:
                    sr = fs
                elif fs != sr:
                    self.q.put(("run_error",
                                f"{f} is {fs} Hz but the others are {sr} Hz. "
                                "Re-export your stimuli at one rate.")); return

            onsets = sched["onset_s"].tolist()
            total_s = float(sched["offset_s"].iloc[-1] + sched["gap_after_s"].iloc[-1])
            N = len(sched)
            t0 = time.perf_counter()
            i, cur = 0, ""
            while True:
                if self.stop_event.is_set():
                    sd.stop()
                    self.q.put(("run_stop", (i, N))); break
                elapsed = time.perf_counter() - t0
                while i < N and elapsed >= onsets[i]:
                    r = sched.iloc[i]
                    sd.play(waves[r["file"]], int(sr))
                    cur = r["file"]
                    self.log.append(dict(trial=int(r["trial"]), file=r["file"],
                                         sound_index=-1, onset_s=round(elapsed, 4)))
                    i += 1
                if i >= N and elapsed >= total_s:
                    self.q.put(("run_done", (N, total_s))); break
                self.q.put(("run_progress", (elapsed, total_s, i, N, cur)))
                time.sleep(0.02)
            pd.DataFrame(self.log).to_csv(os.path.join(OUTPUT_DIR, "trigger_log.csv"), index=False)
        except Exception as e:
            self.q.put(("run_error", f"{e}"))

    def _poll_run(self):
        done = False
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "run_progress":
                    elapsed, total, i, N, cur = payload
                    self.progress["value"] = min(100, elapsed / total * 100)
                    mm, ss = divmod(int(elapsed), 60)
                    tm, ts = divmod(int(total), 60)
                    what = f"playing {cur}" if cur else "starting"
                    self.status_lbl.config(text=f"Sound {i}/{N}   {mm}:{ss:02d} / {tm}:{ts:02d}   {what}")
                elif kind == "run_done":
                    self.progress["value"] = 100
                    self.status_lbl.config(text="Finished. Stop the Open Ephys recording now.")
                    self._log_event("run_finished", sounds_played=len(self.log))
                    done = True
                elif kind == "run_stop":
                    i, N = payload
                    self.status_lbl.config(text=f"Stopped at sound {i}/{N}. Sound off. Stop the recording.")
                    self._log_event("run_stopped", sounds_played=len(self.log))
                    done = True
                elif kind == "run_error":
                    self.status_lbl.config(text="Error: " + str(payload))
                    messagebox.showerror("Problem running experiment", str(payload))
                    done = True
        except queue.Empty:
            pass
        if done:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
        else:
            self.root.after(100, self._poll_run)

    # ---------- Tab 3: analyze ----------
    def _build_analyze(self, tab):
        ttk.Label(tab, text="After a recording", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 2))

        rf = ttk.Frame(tab)
        rf.pack(fill="x", pady=2)
        ttk.Label(rf, text="Recording:").pack(side="left")
        self.v_recording = tk.StringVar(value="")
        ttk.Entry(rf, textvariable=self.v_recording).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(rf, text="Latest", command=self.use_latest_recording).pack(side="left")
        ttk.Button(rf, text="Browse...", command=self.browse_recording).pack(side="left", padx=(4, 0))
        ttk.Label(tab, text="(leave blank to use your most recent recording)",
                  foreground="gray").pack(anchor="w")

        ttk.Label(tab, text="Step 1 - find the sounds", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Button(tab, text="Find sound onsets", command=self.find_onsets).pack(anchor="w", pady=2)
        self.an_box = tk.Text(tab, height=13, width=76, state="disabled")
        self.an_box.pack(fill="x", pady=2)

        ttk.Label(tab, text="Step 2 - sort the spikes (uses the GPU)",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        sfr = ttk.Frame(tab)
        sfr.pack(fill="x", pady=2)
        ttk.Label(sfr, text="Probe:").pack(side="left")
        self.v_probe = tk.StringVar(value="probe_H5D.json")
        ttk.Combobox(sfr, textvariable=self.v_probe, width=18,
                     values=["probe_H5D.json", "probe_H15.json"]).pack(side="left", padx=6)
        ttk.Button(sfr, text="Run spike sorting (Kilosort4)", command=self.run_sorting).pack(side="left")
        self.sort_lbl = ttk.Label(tab, text="", foreground="gray")
        self.sort_lbl.pack(anchor="w")

        ttk.Button(tab, text="No real data yet? Generate demo data matching your experiment",
                   command=self.generate_demo_data).pack(anchor="w", pady=(2, 0))

        self.v_ks_advanced = tk.BooleanVar(value=False)
        self._ks_adv_chk = ttk.Checkbutton(
            tab, variable=self.v_ks_advanced, command=self._toggle_ks_params,
            text="Advanced: change sorting parameters (initial tuning only - keep them the same for a real "
                 "analysis)")
        self._ks_adv_chk.pack(anchor="w", pady=(4, 0))
        self.ks_params = {k: tk.StringVar(value=str(d)) for k, d, _c, _l, _ds in KS_PARAMS}
        self.ks_param_frame = ttk.Frame(tab)
        ttk.Label(self.ks_param_frame, foreground="#b45309", justify="left", wraplength=800,
                  text="Use the SAME parameters for every recording of an experiment. These are only for "
                       "initial tuning; otherwise the defaults are always used.").pack(anchor="w", pady=(2, 6))
        for key, default, cast, lbl, desc in KS_PARAMS:
            row = ttk.Frame(self.ks_param_frame)
            row.pack(fill="x", pady=(4, 0))
            ttk.Label(row, text=lbl + ":", width=30).pack(side="left")
            ttk.Entry(row, textvariable=self.ks_params[key], width=10).pack(side="left")
            ttk.Label(self.ks_param_frame, text=desc, foreground="gray", justify="left",
                      wraplength=800).pack(anchor="w", padx=(12, 0))
        ttk.Button(self.ks_param_frame, text="Reset to defaults",
                   command=self._reset_ks_params).pack(anchor="w", pady=(8, 0))

        ttk.Label(tab, text="Step 3 - view a neuron's response",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        pfr = ttk.Frame(tab)
        pfr.pack(fill="x", pady=2)
        ttk.Label(pfr, text="Unit number:").pack(side="left")
        self.v_unit = tk.StringVar(value="0")
        ttk.Entry(pfr, textvariable=self.v_unit, width=6).pack(side="left", padx=6)
        ttk.Button(pfr, text="Show response (PSTH)", command=self.show_psth).pack(side="left")
        ttk.Button(pfr, text="List units", command=self.list_units).pack(side="left", padx=(4, 0))
        self.unit_box = tk.Text(tab, height=12, width=76, state="disabled")
        self.unit_box.pack(fill="x", pady=2)

        ttk.Label(tab, text="Step 4 - keep going in Phy or Jupyter",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        efr = ttk.Frame(tab)
        efr.pack(fill="x", pady=2)
        ttk.Button(efr, text="Open in Phy (curate units)", command=self.open_phy).pack(side="left")
        ttk.Button(efr, text="Open in Jupyter (custom analysis)",
                   command=self.open_jupyter).pack(side="left", padx=(6, 0))
        self.export_lbl = ttk.Label(tab, text="", foreground="gray")
        self.export_lbl.pack(anchor="w")

    def use_latest_recording(self):
        r = self._find_latest_recording()
        self.v_recording.set(r or "")

    def browse_recording(self):
        d = filedialog.askdirectory(title="Choose an Open Ephys recording folder")
        if d:
            self.v_recording.set(d)

    def _find_latest_recording(self):
        best, bt = None, -1.0
        for root in OE_ROOTS:
            if not os.path.isdir(root):
                continue
            for oeb in glob.glob(os.path.join(root, "**", "structure.oebin"), recursive=True):
                m = os.path.getmtime(oeb)
                if m > bt:
                    bt, best = m, os.path.dirname(oeb)
        return best

    def _resolve_recording(self):
        r = self.v_recording.get().strip() if hasattr(self, "v_recording") else ""
        return r if r else self._find_latest_recording()

    def _find_neural_stream(self, recdir):
        oeb = None
        for dp, _, fn in os.walk(recdir or ""):
            if "structure.oebin" in fn:
                oeb = os.path.join(dp, "structure.oebin")
        if oeb is None:
            return None
        meta = json.load(open(oeb)); base = os.path.dirname(oeb)
        for s in meta["continuous"]:
            nm = [c["channel_name"] for c in s["channels"]]
            neural = [i for i, x in enumerate(nm) if not (x.startswith("ADC") or x == "MEM")]
            if len(neural) >= 16:
                dat = os.path.join(base, "continuous", s["folder_name"], "continuous.dat")
                if not os.path.exists(dat):
                    g = glob.glob(os.path.join(base, "**", "continuous.dat"), recursive=True)
                    dat = g[0] if g else None
                return dict(dat=dat, n_chan=int(s["num_channels"]), fs=float(s["sample_rate"]))
        return None

    def _toggle_ks_params(self):
        if self.v_ks_advanced.get():
            if not messagebox.askyesno("Change sorting parameters?",
                    "Sorting parameters should stay the SAME for every recording of a given experiment.\n\n"
                    "Only change them while first tuning, then keep them fixed for your real analysis.\n\n"
                    "Show the parameters?"):
                self.v_ks_advanced.set(False)
                return
            self.ks_param_frame.pack(fill="x", after=self._ks_adv_chk, pady=(2, 0))
        else:
            self.ks_param_frame.pack_forget()

    def _reset_ks_params(self):
        for k, v in KS_PARAM_DEFAULTS.items():
            self.ks_params[k].set(str(v))

    def run_sorting(self):
        rec = self._resolve_recording()
        if not rec:
            messagebox.showwarning("No recording", "Choose a recording (or record one) first.")
            return
        try:
            params = {}
            for key, default, cast, lbl, desc in KS_PARAMS:
                v = self.ks_params[key].get().strip()
                params[key] = int(float(v)) if cast is int else float(v)
        except ValueError:
            messagebox.showwarning("Bad parameter", "Sorting parameters must be numbers.")
            return
        self.sort_lbl.config(text="Sorting in progress... please wait. Longer recordings take longer.")
        threading.Thread(target=self._sort_worker, args=(rec, params), daemon=True).start()
        self.root.after(200, self._poll_sort)

    def _sort_worker(self, rec, params):
        try:
            ns = self._find_neural_stream(rec)
            if ns is None:
                self.sortq.put("No neural/probe channels in this recording (it may be ADC-only).")
                return
            import torch
            from probeinterface import read_probeinterface
            from kilosort import run_kilosort
            pf = self.v_probe.get()
            if not os.path.isabs(pf):
                pf = os.path.join(HERE, pf)
            pr = read_probeinterface(pf).probes[0]
            pos = pr.contact_positions; dev = pr.device_channel_indices; sh = pr.shank_ids
            probe_ks = dict(chanMap=np.asarray(dev, dtype="int32"),
                            xc=pos[:, 0].astype("float32"), yc=pos[:, 1].astype("float32"),
                            kcoords=(np.asarray(sh, dtype=int).astype("float32")
                                     if sh is not None else np.zeros(len(pos), "float32")),
                            n_chan=int(ns["n_chan"]))
            ks = os.path.join(rec, "kilosort4")
            run_kilosort(settings=dict(n_chan_bin=int(ns["n_chan"]), fs=ns["fs"], **params),
                         probe=probe_ks, filename=ns["dat"], results_dir=ks,
                         data_dtype="int16",
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.ks_dir = ks
            clu = np.load(os.path.join(ks, "spike_clusters.npy"))
            self._log_event("sorting_done", recording=rec, n_units=int(len(np.unique(clu))),
                            parameters=params)
            self.sortq.put(f"Sorting done: {len(np.unique(clu))} units found. Saved to {ks}")
        except Exception as e:
            self.sortq.put(f"Sorting problem: {e}")

    def generate_demo_data(self):
        if self.schedule is None:
            messagebox.showwarning("Design first",
                                   "Build a play order on tab 4 first, so the demo can match your experiment.")
            return
        if not messagebox.askyesno("Generate demo data?",
                "This creates a FAKE recording that matches your current experiment (same sounds and times), "
                "with simulated neurons - some that respond to the sounds. Use it to try the analysis before "
                "you have real data.\n\nIt writes a large file (hundreds of MB to a few GB). Continue?"):
            return
        self.sort_lbl.config(text="Generating demo data... please wait (this writes a large file).")
        threading.Thread(target=self._demo_worker,
                         args=(self.schedule.copy(), self.stimuli_dir, self.v_probe.get()),
                         daemon=True).start()
        self.root.after(200, self._poll_sort)

    def _demo_worker(self, sched, stimuli_dir, probe_file):
        try:
            from math import gcd
            from scipy.signal import resample_poly
            from probeinterface import read_probeinterface
            fs = 30000
            pf = probe_file if os.path.isabs(probe_file) else os.path.join(HERE, probe_file)
            pr = read_probeinterface(pf).probes[0]
            pos = pr.contact_positions
            chanmap = np.asarray(pr.device_channel_indices, dtype=int)
            n_neural = len(pos)
            adc_names = ["ADC1", "ADC2", "ADC3", "ADC4", "ADC5", "ADC6", "ADC7", "ADC8"]
            n_adc = len(adc_names); n_ch = n_neural + n_adc
            total_s = float(sched["offset_s"].iloc[-1] + sched["gap_after_s"].iloc[-1])
            n_samp = int(total_s * fs)
            rng = np.random.default_rng(0)

            stims = {}
            for f in sched["file"].unique():
                x, sr = sf.read(os.path.join(stimuli_dir, f), dtype="float64", always_2d=True)
                x = x[:, 0]
                if int(sr) != fs:
                    g = gcd(fs, int(sr)); x = resample_poly(x, fs // g, int(sr) // g)
                stims[f] = x

            REC = os.path.join(OUTPUT_DIR, "demo_recording", "Record Node 000", "experiment1", "recording1")
            os.makedirs(os.path.join(REC, "continuous", "demo"), exist_ok=True)
            datpath = os.path.join(REC, "continuous", "demo", "continuous.dat")
            master = np.memmap(datpath, dtype=np.int16, mode="w+", shape=(n_samp, n_ch))

            blk = fs * 10
            for t0 in range(0, n_samp, blk):
                t1 = min(n_samp, t0 + blk)
                master[t0:t1, :n_neural] = rng.normal(0, 45, size=(t1 - t0, n_neural)).astype(np.int16)

            tw = np.arange(-20, 40)
            template = -np.exp(-0.5 * (tw / 3.0) ** 2) + 0.4 * np.exp(-0.5 * ((tw - 12) / 6.0) ** 2)
            template /= np.max(np.abs(template))
            onsets_s = sched["onset_s"].values
            units = []
            for j in range(10):
                ci = int(rng.integers(n_neural)); d = np.sqrt(((pos - pos[ci]) ** 2).sum(1))
                units.append(dict(chan_amp=float(rng.uniform(320, 650)) * np.exp(-d / 30.0),
                                  rate=float(rng.uniform(3, 8)), auditory=j < 3))

            def stamp(s0, camp):
                if s0 - 20 < 0 or s0 + 40 > n_samp:
                    return
                for ci in np.where(camp > 8)[0]:
                    ch = chanmap[ci]
                    seg = master[s0 - 20:s0 + 40, ch].astype(np.int32) + (camp[ci] * template).astype(np.int32)
                    master[s0 - 20:s0 + 40, ch] = np.clip(seg, -32000, 32000).astype(np.int16)

            for u in units:
                times = list(rng.uniform(0.05, total_s - 0.05, size=rng.poisson(u["rate"] * total_s)))
                if u["auditory"]:
                    for o in onsets_s:
                        for lat in rng.normal(0.12, 0.03, size=rng.poisson(4)):
                            tt = o + max(0.03, float(lat))
                            if 0 < tt < total_s:
                                times.append(tt)
                for t in np.sort(times):
                    stamp(int(t * fs), u["chan_amp"])

            gate = np.zeros(n_samp, np.int16); audio = np.zeros(n_samp, np.int16)
            for o, f in zip(onsets_s, sched["file"].values):
                s0 = int(o * fs); st = stims[f]; e = min(n_samp, s0 + len(st))
                gate[s0:e] = 6000
                audio[s0:e] = (np.clip(st[:e - s0], -1, 1) * 8000).astype(np.int16)
            master[:, n_neural] = gate; master[:, n_neural + 1] = audio
            for c in range(2, n_adc):
                master[:, n_neural + c] = rng.normal(0, 20, size=n_samp).astype(np.int16)
            master.flush(); del master

            channels = [dict(channel_name=f"CH{i + 1}", bit_volts=0.195) for i in range(n_neural)]
            channels += [dict(channel_name=nm, bit_volts=0.00015258789) for nm in adc_names]
            oebin = dict(GUI_version="demo",
                         continuous=[dict(folder_name="demo/", sample_rate=fs, source_processor_name="Demo",
                                          source_processor_id=0, num_channels=n_ch, channels=channels)],
                         events=[], spikes=[])
            json.dump(oebin, open(os.path.join(REC, "structure.oebin"), "w"), indent=2)
            self._log_event("demo_data_generated", recording=REC, n_sounds=int(len(sched)))
            self.sortq.put(("demo_done", REC))
        except Exception as e:
            self.sortq.put(f"Demo data problem: {e}")

    def _poll_sort(self):
        try:
            while True:
                msg = self.sortq.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "demo_done":
                    self.v_recording.set(msg[1])
                    self.sort_lbl.config(
                        text="Demo data ready and selected as the recording. Now try, in order: "
                             "Find sound onsets -> Run spike sorting -> List units / Show PSTH.")
                else:
                    self.sort_lbl.config(text=msg)
                return
        except queue.Empty:
            pass
        self.root.after(200, self._poll_sort)

    def _ensure_onsets(self):
        """Make sure sound onsets are loaded. Uses the ones from this session, or reloads
        them from a saved sound_onsets.csv, so an already-analyzed recording just works."""
        if self.an_onsets is not None and len(self.an_onsets):
            return True
        rec = self._resolve_recording()
        csvp = os.path.join(rec, "sound_onsets.csv") if rec else ""
        if csvp and os.path.exists(csvp):
            df = pd.read_csv(csvp)
            if "onset_sample" in df.columns:
                self.an_onsets = df["onset_sample"].values.astype(int)
                return True
        return False

    def _load_ks(self):
        ks = self._current_ks_dir()   # this session's sort, or an existing kilosort4 folder
        if not ks or not os.path.exists(os.path.join(ks, "spike_times.npy")):
            messagebox.showwarning("No sorted data",
                                   "No sorted data found for this recording.\n\nEither run spike sorting "
                                   "(step 2), or choose a recording that already has a 'kilosort4' folder.")
            return None
        self.ks_dir = ks
        spk = np.load(os.path.join(ks, "spike_times.npy")).ravel().astype(np.int64)
        clu = np.load(os.path.join(ks, "spike_clusters.npy")).ravel()
        return spk, clu

    def list_units(self):
        loaded = self._load_ks()
        if loaded is None:
            return
        spk, clu = loaded
        uids, counts = np.unique(clu, return_counts=True)
        self._ensure_onsets()
        lines = [f"{len(uids)} units found:"]
        if self.an_onsets is not None and len(self.an_onsets):
            fs, on = self.an_fs, self.an_onsets
            for u, c in zip(uids, counts):
                st = spk[clu == u]
                def rate(a, b):
                    return sum(np.sum((st >= o + int(a * fs)) & (st < o + int(b * fs))) for o in on) / len(on) / (b - a)
                d = rate(0.05, 0.25) - rate(-0.25, -0.05)
                lines.append(f"  unit {int(u):3d}: {int(c):5d} spikes   response {d:+.1f} Hz"
                             + ("   <- responds" if d > 3 else ""))
        else:
            lines.append("(run 'Find sound onsets' too, to see which units respond)")
            for u, c in zip(uids, counts):
                lines.append(f"  unit {int(u):3d}: {int(c):5d} spikes")
        self.say(self.unit_box, "\n".join(lines))

    def show_psth(self):
        loaded = self._load_ks()
        if loaded is None:
            return
        if not self._ensure_onsets():
            messagebox.showwarning("Find onsets first",
                                   "Click 'Find sound onsets' (step 1) first, so the sound times are known.")
            return
        try:
            unit = int(self.v_unit.get())
        except ValueError:
            messagebox.showwarning("Unit number", "Enter a whole number for the unit.")
            return
        spk, clu = loaded
        st = spk[clu == unit]
        if len(st) == 0:
            messagebox.showinfo("Empty unit", f"Unit {unit} has no spikes. Try 'List units' to see valid numbers.")
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fs, onsets = self.an_fs, self.an_onsets
        PRE, POST = 0.2, 0.5
        fig = Figure(figsize=(7, 6))
        ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212, sharex=ax1)
        allrel = []
        for row, o in enumerate(onsets):
            rel = (st[(st >= o - PRE * fs) & (st < o + POST * fs)] - o) / fs
            allrel.append(rel)
            ax1.plot(rel, np.full(rel.shape, row, float), "|", color="k", ms=6)
        ax1.axvline(0, color="r"); ax1.set_ylabel("sound (trial)")
        ax1.set_title(f"Unit {unit}: response to sound onset")
        flat = np.concatenate(allrel) if allrel else np.array([])
        ax2.hist(flat, bins=np.arange(-PRE, POST, 0.02), color="#334155"); ax2.axvline(0, color="r")
        ax2.set_xlabel("time from sound onset (s)"); ax2.set_ylabel("spikes")
        fig.tight_layout()
        win = tk.Toplevel(self.root)
        win.title(f"PSTH - unit {unit}")
        cvs = FigureCanvasTkAgg(fig, master=win)
        cvs.draw()
        cvs.get_tk_widget().pack(fill="both", expand=True)

    # ---------- continue in Phy / Jupyter ----------
    def _current_ks_dir(self):
        if self.ks_dir and os.path.exists(os.path.join(self.ks_dir, "spike_times.npy")):
            return self.ks_dir
        rec = self._resolve_recording()
        if not rec:
            return None
        cand = os.path.join(rec, "kilosort4")           # where the app writes it
        if os.path.exists(os.path.join(cand, "spike_times.npy")):
            return cand
        # the notebook writes it deeper (under continuous/...); find it anywhere under the recording
        hits = glob.glob(os.path.join(rec, "**", "spike_times.npy"), recursive=True)
        return os.path.dirname(hits[0]) if hits else None

    def _find_phy(self):
        import shutil
        # 1) explicit override via env var, 2) on PATH, 3) common conda locations
        env = os.environ.get("FEATHER_PHY")
        if env and os.path.exists(env):
            return env
        onpath = shutil.which("phy")
        if onpath:
            return onpath
        for c in (os.path.expanduser(r"~\.conda\envs\phy2\Scripts\phy.exe"),
                  os.path.expanduser(r"~\miniconda3\envs\phy2\Scripts\phy.exe"),
                  os.path.expanduser(r"~\anaconda3\envs\phy2\Scripts\phy.exe"),
                  r"C:\ProgramData\anaconda3\envs\phy2\Scripts\phy.exe",
                  r"C:\ProgramData\miniconda3\envs\phy2\Scripts\phy.exe"):
            if os.path.exists(c):
                return c
        return None

    def open_phy(self):
        ks = self._current_ks_dir()
        if not ks:
            messagebox.showwarning("Sort first", "Run spike sorting first (a kilosort4 folder is needed).")
            return
        phy = self._find_phy()
        if phy is None:
            messagebox.showinfo("Phy not ready",
                                "Phy is still being installed (or isn't installed yet). Try again in a "
                                "few minutes.\n\nWhen ready, this opens your sorted data in Phy for manual "
                                "curation. Your Kilosort output is already Phy-compatible.")
            return
        import subprocess
        subprocess.Popen([phy, "template-gui", "params.py"], cwd=ks)
        self.export_lbl.config(text="Opening Phy in its own window (this can take several seconds)...")

    def open_jupyter(self):
        rec = self._resolve_recording()
        if not rec:
            messagebox.showwarning("No recording", "Choose a recording first.")
            return
        nb = self._write_analysis_notebook(rec, self._current_ks_dir())
        import shutil
        # prefer jupyter-lab from the same environment running this app, else PATH
        _bindir = os.path.dirname(sys.executable)
        jl = os.path.join(_bindir, "Scripts", "jupyter-lab.exe")
        if not os.path.exists(jl):
            jl = os.path.join(_bindir, "jupyter-lab")   # POSIX (conda/venv) layout
        if not os.path.exists(jl):
            jl = shutil.which("jupyter-lab") or shutil.which("jupyter-lab.exe")
        if not jl:
            messagebox.showinfo("Jupyter not found",
                                "Could not find jupyter-lab. Activate the FEATHER environment "
                                "and run 'pip install jupyterlab', then try again.")
            return
        import subprocess
        subprocess.Popen([jl, os.path.basename(nb)], cwd=os.path.dirname(nb))
        self.export_lbl.config(text=f"Opening Jupyter with {os.path.basename(nb)} (opens in your browser)...")

    def _write_analysis_notebook(self, rec, ks):
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        C = []
        C.append(nbf.v4.new_markdown_cell(
            "# Custom analysis\n\n"
            "This notebook is pre-filled with the paths to your recording and its Kilosort\n"
            "output, so you can start analyzing right away. Run the cells top to bottom, then\n"
            "add your own below. (First run 'Find sound onsets' in the app so `sound_onsets.csv` exists.)"))
        C.append(nbf.v4.new_code_cell(
            "import os\n"
            "import numpy as np, pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            f"KS_DIR    = r\"{ks or ''}\"\n"
            f"RECORDING = r\"{rec}\"\n"
            "ONSETS_CSV = os.path.join(RECORDING, 'sound_onsets.csv')\n\n"
            "_p = {}\n"
            "exec(open(os.path.join(KS_DIR, 'params.py')).read(), _p)\n"
            "fs = float(_p.get('sample_rate', 30000))\n\n"
            "spike_times    = np.load(os.path.join(KS_DIR, 'spike_times.npy')).ravel().astype(np.int64)\n"
            "spike_clusters = np.load(os.path.join(KS_DIR, 'spike_clusters.npy')).ravel()\n"
            "units = np.unique(spike_clusters)\n\n"
            "sounds = pd.read_csv(ONSETS_CSV) if os.path.exists(ONSETS_CSV) else None\n"
            "print(f'{len(units)} units, {len(spike_times)} spikes, fs={fs:.0f} Hz')\n"
            "if sounds is None:\n"
            "    print('No sound_onsets.csv yet - run \"Find sound onsets\" in the app first.')\n"
            "else:\n"
            "    print(f'{len(sounds)} sounds. Columns:', list(sounds.columns))\n"
            "sounds.head() if sounds is not None else None"))
        C.append(nbf.v4.new_code_cell(
            "def psth(unit, pre=0.2, post=0.5, sound=None):\n"
            "    \"\"\"Raster + histogram for one unit around sound onsets.\n"
            "    sound = part of a file name to look at just that sound.\"\"\"\n"
            "    rows = sounds if sound is None else sounds[sounds['sound_file'].astype(str).str.contains(sound)]\n"
            "    st = spike_times[spike_clusters == unit]\n"
            "    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,\n"
            "                           gridspec_kw=dict(height_ratios=[3, 1]))\n"
            "    allrel = []\n"
            "    for i, o in enumerate(rows['onset_sample'].values):\n"
            "        rel = (st[(st >= o - pre*fs) & (st < o + post*fs)] - o) / fs\n"
            "        allrel.append(rel); ax[0].plot(rel, np.full(rel.shape, i, float), '|', color='k', ms=5)\n"
            "    ax[0].axvline(0, color='r'); ax[0].set(ylabel='sound', title=f'unit {unit}')\n"
            "    flat = np.concatenate(allrel) if len(allrel) else np.array([])\n"
            "    ax[1].hist(flat, bins=np.arange(-pre, post, 0.02), color='#334155'); ax[1].axvline(0, color='r')\n"
            "    ax[1].set(xlabel='time from onset (s)', ylabel='spikes'); plt.tight_layout(); plt.show()\n\n"
            "if sounds is not None:\n"
            "    psth(unit=int(units[0]))"))
        C.append(nbf.v4.new_code_cell(
            "# Baseline firing rate of every unit (Hz)\n"
            "dur = spike_times.max() / fs\n"
            "for u in units:\n"
            "    print(f'unit {int(u):3d}: {np.sum(spike_clusters==u)/dur:6.2f} Hz')"))
        C.append(nbf.v4.new_markdown_cell(
            "## Add your own analysis below\n\n"
            "Ideas: compare responses across sounds (`psth(unit, sound='Day7')`), compute response\n"
            "latencies, build tuning curves, or load `continuous.dat` for waveform plots.\n"
            "`sounds` has each sound's onset time and identity; `spike_times`/`spike_clusters` are\n"
            "in samples at `fs`."))
        nb["cells"] = C
        out = os.path.join(rec, "analysis.ipynb")
        with open(out, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        return out

    def find_onsets(self):
        self.say(self.an_box, "Looking for the latest recording...")
        threading.Thread(target=self._onset_worker, daemon=True).start()
        self.root.after(50, self._drain_onsets)

    def _onset_worker(self):
        try:
            rec = self._resolve_recording()
            if rec is None:
                self.q.put(("an", "No Open Ephys recording found.")); return
            oebin = glob.glob(os.path.join(rec, "**", "structure.oebin"), recursive=True)[0]
            meta = json.load(open(oebin)); base = os.path.dirname(oebin)
            s = meta["continuous"][0]
            fs = float(s["sample_rate"]); nch = int(s["num_channels"])
            names = [c["channel_name"] for c in s["channels"]]
            dat = glob.glob(os.path.join(base, "**", "continuous.dat"), recursive=True)[0]
            n = os.path.getsize(dat) // (2 * nch)
            data = np.memmap(dat, dtype="<i2", mode="r", shape=(n, nch))
            gate = data[:, names.index("ADC1")].astype(np.float64)
            mid = (np.percentile(gate, 1) + np.percentile(gate, 99)) / 2
            above = gate > mid

            def dedup(idx):
                out, last = [], -10 ** 9
                for r in idx:
                    if r - last > int(0.2 * fs):
                        out.append(int(r)); last = r
                return out

            onsets = dedup(np.where((~above[:-1]) & (above[1:]))[0] + 1)   # rising = sound starts
            offsets = dedup(np.where((above[:-1]) & (~above[1:]))[0] + 1)  # falling = sound ends
            self.an_onsets = np.array(onsets, dtype=int)   # kept for the PSTH step
            self.an_fs = fs

            # Identify each sound directly from the recorded audio (ADC2). This reads identity
            # from the recording itself, so it does not depend on schedule.csv being the right one.
            audio_id = self._identify_by_audio(data, names, fs, onsets) if "ADC2" in names else None

            rows = []
            for k, o in enumerate(onsets):
                off = next((x for x in offsets if x > o), None)
                row = dict(sound_number=k + 1,
                           onset_sample=o, onset_seconds=round(o / fs, 6),
                           offset_sample=(off if off is not None else -1),
                           offset_seconds=(round(off / fs, 6) if off is not None else -1.0),
                           marker_width_seconds=(round((off - o) / fs, 6) if off is not None else -1.0))
                if audio_id is not None:
                    row["sound_from_audio"] = audio_id[k][0]
                    row["audio_offset_ms"] = round(audio_id[k][1], 3)
                    row["audio_corr"] = round(audio_id[k][2], 3)
                rows.append(row)
            tab = pd.DataFrame(rows)

            sched_path = os.path.join(OUTPUT_DIR, "schedule.csv")
            labeled = os.path.exists(sched_path)
            if labeled:
                sched = pd.read_csv(sched_path)
                m = min(len(sched), len(tab))
                tab = tab.iloc[:m].copy()
                tab["sound_from_schedule"] = sched["file"].values[:m]

            # Main identity column: use the audio match when we have it, else the schedule order.
            if audio_id is not None:
                tab["sound_file"] = tab["sound_from_audio"]
            elif labeled:
                tab["sound_file"] = tab["sound_from_schedule"]

            save_rec = os.path.join(rec, "sound_onsets.csv")
            save_out = os.path.join(OUTPUT_DIR, "sound_onsets_latest.csv")
            tab.to_csv(save_rec, index=False)
            tab.to_csv(save_out, index=False)

            lines = [f"Recording: {rec}", f"Length: {n / fs:.1f} s",
                     f"Found {len(onsets)} sound starts and {len(offsets)} sound ends.", ""]
            disagree = 0
            offsets_ms = []
            for _, r in tab.iterrows():
                name = str(r.get("sound_file", "?")).split("_")[0]
                extra = ""
                if audio_id is not None:
                    if float(r.get("audio_corr", 0.0)) < 0.3:
                        extra = "   (no matching audio found - check ADC2 wiring/upload)"
                    else:
                        off = float(r["audio_offset_ms"]); offsets_ms.append(off)
                        extra = f"   audio confirms this sound, {off:+.2f} ms vs marker"
                    if "sound_from_schedule" in tab.columns and \
                            r["sound_from_audio"] != r["sound_from_schedule"]:
                        extra += "  <-- WRONG SOUND vs schedule"; disagree += 1
                lines.append(f"  sound {int(r['sound_number']):2d}: "
                             f"starts {r['onset_seconds']:8.3f} s   {name}{extra}")
            if audio_id is None:
                lines.append("\nNo ADC2 audio channel found, so identity is from the schedule order only.")
            else:
                if offsets_ms:
                    a = np.array(offsets_ms)
                    lines.append(f"\nAudio-vs-marker time offset: mean {a.mean():+.2f} ms, "
                                 f"SD {a.std():.2f} ms, range [{a.min():+.2f}, {a.max():+.2f}] ms "
                                 f"(+ = audio after marker).")
                    lines.append("  The recorded sound sits right on the timing marker (sub-millisecond)."
                                 if np.max(np.abs(a)) < 1.0 else
                                 "  NOTE: offset exceeds 1 ms on some sounds - review before aligning spikes.")
                if disagree:
                    lines.append(f"Warning: on {disagree} sound(s) the played sound does not match "
                                 "schedule.csv - check that schedule.csv is the right one.")
                else:
                    lines.append("Every played sound matches its schedule label.")

            # ---- Timing sync check: the marker width should equal each sound's length ----
            resid_ms, width_s = [], []
            durcache = {}
            for k, o in enumerate(onsets):
                off = next((x for x in offsets if x > o), None)
                if off is None:
                    continue
                w = (off - o) / fs
                width_s.append(w)
                if audio_id is not None:
                    f = audio_id[k][0]
                    if f not in durcache:
                        try:
                            durcache[f] = sf.info(os.path.join(self.stimuli_dir, f)).duration
                        except Exception:
                            durcache[f] = None
                    if durcache[f]:
                        resid_ms.append((w - durcache[f]) * 1000.0)
            lines.append("\nTiming check (sound-to-recording sync):")
            if resid_ms:
                jit = float(np.std(resid_ms))
                lines.append(f"  timing marker matches each sound's length to "
                             f"{np.mean(np.abs(resid_ms)):.3f} ms on average, jitter (SD) {jit:.3f} ms")
                lines.append("  WARNING: jitter above 1 ms - check the sync wiring." if jit > 1.0
                             else "  Sync is sample-accurate (jitter well under 1 ms). Good to use.")
            elif width_s:
                jit = float(np.std(width_s) * 1000.0)
                lines.append(f"  marker width {np.mean(width_s):.4f} s, jitter (SD) {jit:.3f} ms")
                lines.append("  WARNING: jitter above 1 ms - check the sync." if jit > 1.0
                             else "  Sync is sample-accurate. Good to use.")
            else:
                lines.append("  could not measure (no complete marker pairs).")

            # ---- Optional microphone-at-the-bird calibration ----
            if MIC_CHANNEL and MIC_CHANNEL in names:
                mic = data[:, names.index(MIC_CHANNEL)].astype(np.float64)
                wv = max(1, int(0.0005 * fs))
                menv = np.convolve(np.abs(mic - np.median(mic)), np.ones(wv) / wv, mode="same")
                delays = []
                for o in onsets:
                    pre = menv[max(0, o - int(0.03 * fs)):o]
                    if len(pre) < 2:
                        continue
                    thr = np.median(pre) + 8 * (np.std(pre) + 1e-9)
                    hit = np.where(menv[o:o + int(0.2 * fs)] > thr)[0]
                    if len(hit):
                        delays.append(hit[0] / fs * 1000.0)
                enough = len(delays) >= max(2, len(onsets) // 2)
                if enough and float(np.std(delays)) < 20.0:
                    d = np.array(delays)
                    lines.append(f"\nMicrophone ({MIC_CHANNEL}): sound reaches the bird "
                                 f"{d.mean():.2f} ms after the marker (SD {d.std():.2f} ms).")
                    lines.append(f"  For true sound-at-bird times, add {d.mean():.2f} ms to the onset times.")
                elif enough:
                    lines.append(f"\nMicrophone ({MIC_CHANNEL}): signal found but delays are inconsistent "
                                 f"(SD {np.std(delays):.1f} ms) - check the mic and its placement.")
                else:
                    lines.append(f"\nMicrophone ({MIC_CHANNEL}): no clear signal - no mic connected there yet. "
                                 "Wire a mic to that channel to record the true sound-at-bird timing.")

            lines += ["", "Saved (start time, end time, and which sound):", "  " + save_rec, "  " + save_out]
            self._log_event("onsets_found", recording=rec, n_onsets=int(len(onsets)))
            self.q.put(("an", "\n".join(lines)))
        except Exception as e:
            self.q.put(("an", f"Problem: {e}"))
        self.root.after(50, self._drain_onsets)

    def _identify_by_audio(self, data, names, fs, onsets):
        """For each onset, find which stimulus WAV the recorded ADC2 audio matches, using
        a lag-aligned cross-correlation, and measure the TIME OFFSET (in ms) between the
        timing marker and the recorded audio. Returns (best_file, offset_ms, corr) per
        onset: offset_ms > 0 means the audio arrives AFTER the marker; corr (0..1) is only
        used internally to flag events where no real sound was found."""
        from math import gcd
        from scipy.signal import resample_poly, correlate
        fsi = int(round(fs))
        audio = data[:, names.index("ADC2")].astype(np.float64)
        stims = {}
        for f in sorted(os.listdir(self.stimuli_dir)):
            if f.lower().endswith(".wav"):
                x, sr = sf.read(os.path.join(self.stimuli_dir, f), dtype="float64", always_2d=True)
                x = x[:, 0]
                sr = int(sr)
                if sr != fsi:
                    g = gcd(fsi, sr)
                    x = resample_poly(x, fsi // g, sr // g)
                stims[f] = x - x.mean()          # keep native scale; normalize per-comparison
        seglen = int(5.2 * fs)
        maxlag = int(0.060 * fs)                 # only trust offsets within +-60 ms of the marker
        out = []
        for o in onsets:
            seg = audio[o:o + seglen].astype(np.float64)
            seg = seg - seg.mean()
            best_f, best_c, best_lag = None, -1e9, 0.0
            for f, t in stims.items():
                L = min(len(seg), len(t))
                a = seg[:L] / (np.linalg.norm(seg[:L]) + 1e-9)
                b = t[:L] / (np.linalg.norm(t[:L]) + 1e-9)
                c = correlate(a, b, mode="full")           # peak = lag that aligns audio to template
                lags = np.arange(-(L - 1), L)
                cc = np.where(np.abs(lags) <= maxlag, c, -np.inf)
                i = int(np.argmax(cc))
                if c[i] > best_c:
                    delta = 0.0                            # parabolic sub-sample refinement
                    if 0 < i < len(c) - 1:
                        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
                        den = y0 - 2 * y1 + y2
                        if den != 0:
                            delta = 0.5 * (y0 - y2) / den
                    best_c = float(c[i])
                    best_f = f
                    best_lag = (lags[i] + delta) / fs * 1000.0
            out.append((best_f, best_lag, best_c))
        return out

    def _drain_onsets(self):
        try:
            while True:
                kind, text = self.q.get_nowait()
                if kind == "an":
                    self.say(self.an_box, text)
        except queue.Empty:
            pass


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
