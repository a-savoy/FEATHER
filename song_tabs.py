"""song_tabs.py — the Record and Sound Library tabs (Sound Bench).

These are mixed into the FEATHER app (see playback_gui.py). They are kept in their own
file only so playback_gui.py stays readable; they follow exactly the same patterns as
the tabs already there — worker thread, queue, and a periodic poll on the Tk main loop,
so the window never freezes while something long is running.

Tab 1 (Record Vox)     drives song_recorder: levels, live sonogram, keep/discard log.
Tab 2 (Generate Calls) drives song_generate: the VAE's sliders, preview, and saving.
Tab 3 (Sound Library)  drives song_library + song_features: browse everything you have
                       recorded or generated, look at the sonogram and features, tune
                       segmentation, and cut a clip into the stimulus folder that
                       Design Experiment reads.

The order follows the work: make sounds (record real ones or generate new ones), collect
them in the library, then design an experiment with them.
"""
import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

import song_features as sfx
import song_library as slib
import song_recorder as srec
import song_generate as sgen

MONITOR_SECONDS = 4.0        # width of the live sonogram
PAGE_SECONDS = 20.0          # how much of a long recording is drawn at once


class SoundBenchTabs:
    """Mixin providing the Record and Sound Library tabs."""

    # ================================================================== state
    def _sb_init(self):
        self.sb_recorder = None
        self.sb_recq = queue.Queue()
        self.sb_monitor_buf = None
        self.sb_lib = None
        self.sb_scan = None            # DataFrame of found recordings
        self.sb_table = None           # syllable table for the whole folder
        self.sb_libq = queue.Queue()
        self.sb_stop_scan = threading.Event()
        self.sb_path = None            # file currently in the viewer
        self.sb_audio = None
        self.sb_fs = None
        self.sb_feat = None
        self.sb_spec = None
        self.sb_syl = None
        self.sb_page = 0.0
        self.sb_sel = None             # (start_s, end_s) selection in the viewer
        self.audio_playing = None      # label of whatever is sounding right now
        self.gen_vars = {}             # generator sliders
        self.gen_audio = None
        self.gen_q = queue.Queue()
        self.gen_busy = False

    def _sb_library(self):
        if self.sb_lib is None:
            self.sb_lib = slib.Library(os.path.join(HERE_SB, slib.CACHE_NAME))
        return self.sb_lib

    def _sb_spec_params(self):
        try:
            return dict(f_low=float(self.v_sb_flow.get()), f_high=float(self.v_sb_fhigh.get()))
        except Exception:
            return {}

    def _sb_display_params(self):
        """The frequency range the sonogram is DRAWN over — not what is measured."""
        try:
            return dict(f_low=float(self.v_sb_dlow.get()),
                        f_high=float(self.v_sb_dhigh.get()))
        except Exception:
            return dict(sfx.DISPLAY_DEFAULTS)

    def _sb_display_spec(self, x, fs):
        return sfx.spectral(np.asarray(x, dtype=float), fs, **self._sb_display_params())

    def _sb_segment_params(self):
        p = {}
        try:
            p["threshold_below_peak_db"] = float(self.v_sb_thresh.get())
            p["min_duration_ms"] = float(self.v_sb_mindur.get())
            p["min_gap_ms"] = float(self.v_sb_mingap.get())
            p["feature"] = self.v_sb_segfeat.get()
            p["noise_highpass_hz"] = float(self.v_sb_noise.get())
        except Exception:
            pass
        return p

    # ================================================= shared audio auditioning
    # Every tab that can make a sound plays it through here, so one Stop button
    # anywhere silences whatever is sounding. This is the computer's speakers, not
    # the Harp card - it is for choosing stimuli, not for running an experiment.
    def _audio_play(self, y, sr, label=""):
        try:
            import sounddevice as sd
        except Exception as e:
            messagebox.showwarning("No sound output",
                                   f"Could not use the computer's sound output:\n{e}")
            return False
        try:
            sd.stop()
            sd.play(np.ascontiguousarray(np.asarray(y, dtype="float32")), int(sr))
        except Exception as e:
            messagebox.showwarning("Could not play", str(e))
            return False
        self.audio_playing = label
        self._audio_note(f"Playing {label}" if label else "Playing")
        return True

    def _audio_play_file(self, path):
        try:
            y, fs = slib.read_audio(path)
        except Exception as e:
            messagebox.showwarning("Could not read that sound", str(e))
            return
        self._audio_play(y, fs, os.path.basename(path))

    def _audio_stop(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.audio_playing = None
        self._audio_note("Stopped.")

    def _audio_note(self, text):
        for attr in ("stim_play_lbl", "gen_status", "sb_lib_status"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.config(text=text)
                except Exception:
                    pass


    # ============================================================ TAB: RECORD
    def _build_record(self, tab):
        ttk.Label(tab, text="Record vocalizations",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6, 0))
        ttk.Label(tab, wraplength=820, justify="left", foreground="gray",
                  text="Listens on each channel and saves a file whenever a bird calls. "
                       "Recording opens a couple of seconds BEFORE the sound, so onsets are "
                       "never clipped. Each file is then checked to make sure it really is a "
                       "vocalization; anything rejected is moved to a 'rejected' folder with "
                       "the reason, never deleted.").pack(anchor="w", pady=(2, 8))

        # ---- device ----
        dev = ttk.LabelFrame(tab, text="Microphone")
        dev.pack(fill="x", pady=4)
        row = ttk.Frame(dev); row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Input device:").pack(side="left")
        self.v_sb_device = tk.StringVar()
        self.sb_device_box = ttk.Combobox(row, textvariable=self.v_sb_device,
                                          width=46, state="readonly")
        self.sb_device_box.pack(side="left", padx=6)
        ttk.Button(row, text="Refresh", command=self._sb_refresh_devices).pack(side="left")

        row2 = ttk.Frame(dev); row2.pack(fill="x", padx=8, pady=(0, 8))
        self.v_sb_channels = tk.StringVar(value="1")
        self.v_sb_fs = tk.StringVar(value="44100")
        self.v_sb_birds = tk.StringVar(value="bird1")
        ttk.Label(row2, text="Channels:").pack(side="left")
        ttk.Entry(row2, textvariable=self.v_sb_channels, width=4).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="Sample rate:").pack(side="left")
        ttk.Entry(row2, textvariable=self.v_sb_fs, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="Bird names (one per channel, commas):").pack(side="left")
        ttk.Entry(row2, textvariable=self.v_sb_birds, width=24).pack(side="left", padx=4)

        # ---- destination ----
        dest = ttk.Frame(tab); dest.pack(fill="x", pady=4)
        ttk.Label(dest, text="Save recordings in:").pack(side="left")
        self.v_sb_outdir = tk.StringVar(value=os.path.join(HERE_SB, "song_data"))
        ttk.Entry(dest, textvariable=self.v_sb_outdir).pack(side="left", fill="x",
                                                            expand=True, padx=6)
        ttk.Button(dest, text="Browse...", command=self._sb_browse_outdir).pack(side="left")

        # ---- when to record ----
        trg = ttk.LabelFrame(tab, text="When to record")
        trg.pack(fill="x", pady=6)
        self.v_sb_mode = tk.StringVar(value="triggered")
        mrow = ttk.Frame(trg); mrow.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Radiobutton(mrow, text="Only when a bird calls (recommended)",
                        variable=self.v_sb_mode, value="triggered").pack(side="left")
        ttk.Radiobutton(mrow, text="Continuously, in files of",
                        variable=self.v_sb_mode, value="continuous").pack(side="left", padx=(16, 4))
        self.v_sb_contlen = tk.StringVar(value="60")
        ttk.Entry(mrow, textvariable=self.v_sb_contlen, width=5).pack(side="left")
        ttk.Label(mrow, text="seconds").pack(side="left", padx=(4, 0))

        g = ttk.Frame(trg); g.pack(fill="x", padx=8, pady=(2, 8))
        self.v_sb_thresh_db = tk.StringVar(value="-40")
        self.v_sb_hold = tk.StringVar(value="1.0")
        self.v_sb_prebuf = tk.StringVar(value="2.0")
        self.v_sb_minev = tk.StringVar(value="2")
        self._grid_row(g, 0, "Start recording above (dB):", self.v_sb_thresh_db,
                       hint="-40 is a good starting point; watch the meters below")
        self._grid_row(g, 1, "Stop after silence (seconds):", self.v_sb_hold, hint="1.0 like SAP")
        self._grid_row(g, 2, "Keep sound from before (seconds):", self.v_sb_prebuf,
                       hint="the pre-buffer, so onsets are not clipped")
        self._grid_row(g, 3, "Needs at least this many calls:", self.v_sb_minev,
                       hint="rejects a single door slam")

        chk = ttk.LabelFrame(tab, text="Check each recording is really a vocalization")
        chk.pack(fill="x", pady=4)
        self.v_sb_check = tk.BooleanVar(value=True)
        ttk.Checkbutton(chk, variable=self.v_sb_check,
                        text="After each file, measure it and discard non-vocal noise"
                        ).pack(anchor="w", padx=8, pady=(6, 0))
        g2 = ttk.Frame(chk); g2.pack(fill="x", padx=8, pady=(2, 8))
        self.v_sb_minsyl = tk.StringVar(value="3")
        self.v_sb_minbout = tk.StringVar(value="150")
        self._grid_row(g2, 0, "Fewest calls in the file:", self.v_sb_minsyl,
                       hint="begging bouts are repetitive; 3 is a sensible floor")
        self._grid_row(g2, 1, "Shortest bout (ms):", self.v_sb_minbout,
                       hint="150 ms suits begging calls; song needs more")

        # ---- controls + meters ----
        ctl = ttk.Frame(tab); ctl.pack(fill="x", pady=(8, 2))
        self.sb_start_btn = ttk.Button(ctl, text="Start recording", command=self._sb_start_record)
        self.sb_start_btn.pack(side="left")
        self.sb_stop_btn = ttk.Button(ctl, text="Stop", command=self._sb_stop_record,
                                      state="disabled")
        self.sb_stop_btn.pack(side="left", padx=6)
        self.sb_rec_status = ttk.Label(ctl, text="Not recording.", foreground="gray")
        self.sb_rec_status.pack(side="left", padx=10)

        self.sb_meter_frame = ttk.Frame(tab)
        self.sb_meter_frame.pack(fill="x", pady=4)
        self.sb_meters = []

        ttk.Label(tab, text="Live sound", font=("Segoe UI", 9, "bold")).pack(anchor="w",
                                                                            pady=(6, 0))
        self.sb_monitor_holder = ttk.Frame(tab, height=150)
        self.sb_monitor_holder.pack(fill="x", pady=2)
        self.sb_monitor_canvas = None

        ttk.Label(tab, text="Recordings kept and rejected",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        self.sb_rec_box = tk.Text(tab, height=9, width=100, state="disabled")
        self.sb_rec_box.pack(fill="both", expand=True, pady=2)

        self._sb_refresh_devices()
        self.root.after(150, self._sb_poll_record)

    def _sb_refresh_devices(self):
        try:
            devs = srec.list_devices()
        except Exception as e:
            self.sb_device_box["values"] = []
            self.say(self.sb_rec_box, f"Could not list audio devices: {e}")
            return
        self.sb_devices = devs
        labels = [f"{i}: {name} ({ch} in)" for i, name, ch, _sr in devs]
        self.sb_device_box["values"] = labels
        if labels and not self.v_sb_device.get():
            self.sb_device_box.current(0)

    def _sb_browse_outdir(self):
        d = filedialog.askdirectory(title="Folder to save recordings in",
                                    initialdir=self.v_sb_outdir.get() or HERE_SB)
        if d:
            self.v_sb_outdir.set(d)

    def _sb_selected_device(self):
        label = self.v_sb_device.get()
        if not label:
            return None
        try:
            return int(label.split(":")[0])
        except Exception:
            return None

    def _sb_build_meters(self, names):
        for w in self.sb_meter_frame.winfo_children():
            w.destroy()
        self.sb_meters = []
        for n in names:
            row = ttk.Frame(self.sb_meter_frame); row.pack(fill="x", pady=1)
            ttk.Label(row, text=n, width=12).pack(side="left")
            bar = ttk.Progressbar(row, maximum=60, length=320)
            bar.pack(side="left", padx=6)
            lvl = ttk.Label(row, text="-- dB", width=9)
            lvl.pack(side="left")
            rec = ttk.Label(row, text="", width=12, foreground="#b91c1c")
            rec.pack(side="left")
            self.sb_meters.append((bar, lvl, rec))

    def _sb_start_record(self):
        if self.sb_recorder is not None:
            return
        try:
            n_ch = int(self.v_sb_channels.get())
            fs = float(self.v_sb_fs.get())
            names = [s.strip() for s in self.v_sb_birds.get().split(",") if s.strip()]
            while len(names) < n_ch:
                names.append(f"ch{len(names) + 1}")
            trigger = dict(threshold_db=float(self.v_sb_thresh_db.get()),
                           hold_s=float(self.v_sb_hold.get()),
                           prebuffer_s=float(self.v_sb_prebuf.get()),
                           min_events=int(self.v_sb_minev.get()))
            accept = dict(enabled=bool(self.v_sb_check.get()),
                          min_syllables=int(self.v_sb_minsyl.get()),
                          min_bout_ms=float(self.v_sb_minbout.get()))
        except ValueError as e:
            messagebox.showwarning("Check the settings", f"A recording setting is not a number:\n{e}")
            return

        outdir = self.v_sb_outdir.get().strip() or os.path.join(HERE_SB, "song_data")
        rec = srec.Recorder(outdir=outdir, device=self._sb_selected_device(),
                            channels=n_ch, fs=fs, bird_names=names,
                            trigger=trigger, accept=accept,
                            continuous=(self.v_sb_mode.get() == "continuous"),
                            continuous_s=float(self.v_sb_contlen.get() or 60),
                            on_event=lambda row: self.sb_recq.put(row),
                            spec_params=self._sb_spec_params(),
                            segment_params=self._sb_segment_params())
        try:
            rec.start()
        except Exception as e:
            messagebox.showerror("Could not start recording",
                                 f"{e}\n\nCheck the device, channel count and sample rate. "
                                 "A device may also be in use by another program.")
            return

        srec.write_session(outdir, dict(device=self.v_sb_device.get(), channels=n_ch,
                                        sample_rate=fs, birds=names, trigger=trigger,
                                        accept=accept, started=self.session_id))
        self.sb_recorder = rec
        self._sb_build_meters(names)
        self.sb_start_btn.config(state="disabled")
        self.sb_stop_btn.config(state="normal")
        self.sb_rec_status.config(text="Recording. Waiting for the birds...", foreground="#166534")
        self.say(self.sb_rec_box, f"Listening on {n_ch} channel(s) at {fs:g} Hz.\nSaving to {outdir}\n")
        self._log_event("recording_started", outdir=outdir, channels=n_ch, trigger=trigger)

    def _sb_stop_record(self):
        if self.sb_recorder is None:
            return
        rec = self.sb_recorder
        self.sb_recorder = None
        try:
            rec.stop()
        except Exception:
            pass
        self.sb_start_btn.config(state="normal")
        self.sb_stop_btn.config(state="disabled")
        self.sb_rec_status.config(text=f"Stopped. Kept {rec.n_kept}, rejected {rec.n_rejected}.",
                                  foreground="gray")
        self._log_event("recording_stopped", kept=rec.n_kept, rejected=rec.n_rejected)

    def _sb_poll_record(self):
        rec = self.sb_recorder
        if rec is not None:
            for (bar, lvl, flag), db, on in zip(self.sb_meters, rec.levels(),
                                                rec.recording_flags()):
                bar["value"] = max(0.0, min(60.0, db + 60.0))
                lvl.config(text=f"{db:5.1f} dB")
                flag.config(text="RECORDING" if on else "")
            self._sb_draw_monitor(rec)
            if rec.error:
                self.sb_rec_status.config(text=f"Problem: {rec.error}", foreground="#b91c1c")
        while True:
            try:
                row = self.sb_recq.get_nowait()
            except queue.Empty:
                break
            mark = "kept    " if row["accepted"] else "rejected"
            extra = f"  ({row['reason']})" if row["reason"] else ""
            self._sb_append(self.sb_rec_box,
                            f"{row['started']}  {mark}  {row['bird']}  "
                            f"{row['duration_s']:.2f}s  peak {row['peak_db']:.0f} dB{extra}\n")
        self.root.after(150, self._sb_poll_record)

    def _sb_append(self, box, text):
        box.config(state="normal")
        box.insert("end", text)
        box.see("end")
        box.config(state="disabled")

    def _sb_draw_monitor(self, rec):
        """Scrolling sonogram of the last few seconds, throttled to the poll rate."""
        blocks = []
        while True:
            try:
                blocks.append(rec.monitor.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            return
        new = np.concatenate(blocks)
        keep = int(MONITOR_SECONDS * rec.fs)
        buf = new if self.sb_monitor_buf is None else np.concatenate([self.sb_monitor_buf, new])
        self.sb_monitor_buf = buf[-keep:]
        if len(self.sb_monitor_buf) < int(0.3 * rec.fs):
            return
        try:
            sp = sfx.spectral(self.sb_monitor_buf, rec.fs, advance_ms=4.0,
                              **self._sb_display_params())
        except Exception:
            return
        if len(sp["t"]) < 4:
            return
        ax = self._sb_monitor_axes()
        if ax is None:
            return
        ax.clear()
        img = np.abs(sp["deriv"]).T
        vmax = np.percentile(img, 99.5) or 1.0
        ax.imshow(img, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=vmax,
                  extent=[0, MONITOR_SECONDS, sp["freqs"][0] / 1000, sp["freqs"][-1] / 1000])
        ax.set_ylabel("kHz", fontsize=8)
        ax.set_xlabel("last %g seconds" % MONITOR_SECONDS, fontsize=8)
        ax.tick_params(labelsize=7)
        self.sb_monitor_canvas.draw_idle()

    def _sb_monitor_axes(self):
        if self.sb_monitor_canvas is None:
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            except Exception:
                return None
            fig = Figure(figsize=(8, 1.6), dpi=90)
            self.sb_monitor_ax = fig.add_subplot(111)
            fig.tight_layout()
            self.sb_monitor_canvas = FigureCanvasTkAgg(fig, master=self.sb_monitor_holder)
            self.sb_monitor_canvas.get_tk_widget().pack(fill="both", expand=True)
        return self.sb_monitor_ax

    # =================================================== TAB: GENERATE CALLS
    def _build_generate(self, tab):
        ttk.Label(tab, text="Generate begging calls",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6, 0))
        ok, why = sgen.available()
        ranges = sgen.empirical_ranges()
        counts = ranges.get("counts", {})
        blurb = ("Makes new begging calls with a model trained on "
                 f"{counts.get('n_calls', 0):,} real calls from chicks aged 4-15 days. "
                 "Move the sliders, press Listen, and save what you like as a stimulus - "
                 "it then behaves like any recorded sound. Settings outside what real "
                 "chicks do are allowed on purpose, for supernormal stimuli; those are "
                 "marked and written into the stimulus record.")
        ttk.Label(tab, wraplength=820, justify="left", foreground="gray",
                  text=blurb).pack(anchor="w", pady=(2, 8))

        if not ok:
            box = tk.Text(tab, height=6, width=100, state="disabled")
            box.pack(fill="x", pady=6)
            self.say(box, "The call generator is not available:\n\n" + why)
            return

        try:
            self.gen_defaults = sgen.defaults()
        except Exception as e:
            ttk.Label(tab, foreground="#b91c1c",
                      text=f"Could not load the generator: {e}").pack(anchor="w")
            return

        specs = {sp[0]: sp for sp in sgen.PARAM_SPECS}
        for group_name, keys in sgen.GROUPS:
            frame = ttk.LabelFrame(tab, text=group_name)
            frame.pack(fill="x", pady=4)
            for row, key in enumerate(keys):
                _k, label, lo, hi, step, typical, help_text = specs[key]
                self._gen_slider(frame, row, key, label, lo, hi, step, typical, help_text)

        ctl = ttk.Frame(tab); ctl.pack(fill="x", pady=(10, 2))
        ttk.Button(ctl, text="Listen", command=self._gen_preview).pack(side="left")
        ttk.Button(ctl, text="Play again", command=self._gen_replay).pack(side="left", padx=4)
        ttk.Button(ctl, text="Stop", command=self._audio_stop).pack(side="left", padx=(0, 4))
        ttk.Button(ctl, text="Save as stimulus", command=self._gen_save).pack(side="left", padx=6)
        ttk.Button(ctl, text="Reset", command=self._gen_reset).pack(side="left")
        ttk.Button(ctl, text="New seed", command=self._gen_new_seed).pack(side="left", padx=6)
        self.gen_status = ttk.Label(ctl, text="", foreground="gray")
        self.gen_status.pack(side="left", padx=10)

        dest = ttk.Frame(tab); dest.pack(fill="x", pady=2)
        ttk.Label(dest, text="Save into sound folder:").pack(side="left")
        self.v_gen_dest = tk.StringVar(value="")
        ttk.Entry(dest, textvariable=self.v_gen_dest, width=44).pack(side="left", padx=6)
        ttk.Button(dest, text="Use Design tab's folder",
                   command=lambda: self.v_gen_dest.set(self.v_stimdir.get())).pack(side="left")

        # graded series - the way a supernormal experiment is actually built
        ser = ttk.LabelFrame(tab, text="Make a graded set (one stimulus per step)")
        ser.pack(fill="x", pady=(8, 4))
        srow = ttk.Frame(ser); srow.pack(fill="x", padx=8, pady=6)
        ttk.Label(srow, text="Vary").pack(side="left")
        self.v_gen_series_key = tk.StringVar(value="centroid_khz")
        ttk.Combobox(srow, textvariable=self.v_gen_series_key, width=18, state="readonly",
                     values=[sp[0] for sp in sgen.PARAM_SPECS]).pack(side="left", padx=4)
        ttk.Label(srow, text="over").pack(side="left")
        self.v_gen_series_vals = tk.StringVar(value="2.5, 3.5, 4.5, 5.5, 6.5")
        ttk.Entry(srow, textvariable=self.v_gen_series_vals, width=30).pack(side="left", padx=4)
        ttk.Label(srow, text="(commas)", foreground="gray").pack(side="left")
        ttk.Button(srow, text="Generate the set",
                   command=self._gen_series).pack(side="left", padx=10)
        ttk.Label(ser, foreground="gray", wraplength=800, justify="left",
                  text="Everything else stays fixed, including the seed, so the set differs in "
                       "exactly one property. Brightness is the most dependable axis to grade "
                       "on; the pitch slider does not move the measured centre evenly.").pack(
                           anchor="w", padx=8, pady=(0, 6))

        ttk.Label(tab, text="Preview", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        self.gen_view_holder = ttk.Frame(tab, height=220)
        self.gen_view_holder.pack(fill="both", expand=True, pady=2)
        self.gen_canvas = None
        self.gen_box = tk.Text(tab, height=7, width=100, state="disabled")
        self.gen_box.pack(fill="x", pady=4)
        self.say(self.gen_box, "Press Listen to hear and see the call these settings make.")
        self.root.after(200, self._gen_poll)

    def _gen_slider(self, parent, row, key, label, lo, hi, step, typical, help_text):
        var = tk.DoubleVar(value=float(self.gen_defaults.get(key, lo)))
        self.gen_vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 8), pady=2)
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, length=250,
                          command=lambda _v, k=key: self._gen_value_changed(k))
        scale.grid(row=row, column=1, sticky="w", pady=2)
        val = ttk.Label(parent, text="", width=8)
        val.grid(row=row, column=2, sticky="w", padx=(6, 4))
        note = ttk.Label(parent, text="", foreground="gray")
        note.grid(row=row, column=3, sticky="w", padx=(4, 8))
        self.gen_vars[key + "__labels"] = (val, note, typical, key in sgen.INT_PARAMS)
        ttk.Label(parent, text=help_text, foreground="gray", wraplength=760,
                  justify="left").grid(row=row, column=4, sticky="w", padx=(8, 8))
        self._gen_value_changed(key)

    def _gen_value_changed(self, key):
        var = self.gen_vars[key]
        val, note, typical, is_int = self.gen_vars[key + "__labels"]
        v = var.get()
        if is_int:
            v = int(round(v))
        val.config(text=f"{v:g}")
        if typical and sgen.is_supernormal(key, v):
            note.config(text=f"supernormal (typical {typical[0]:g}-{typical[1]:g})",
                        foreground="#b45309")
        elif typical:
            note.config(text=f"typical {typical[0]:g}-{typical[1]:g}", foreground="gray")
        else:
            note.config(text="")

    def _gen_params(self):
        p = dict(self.gen_defaults)
        for key in list(self.gen_vars):
            if key.endswith("__labels"):
                continue
            v = self.gen_vars[key].get()
            p[key] = int(round(v)) if key in sgen.INT_PARAMS else float(v)
        return p

    def _gen_reset(self):
        for key, var in self.gen_vars.items():
            if key.endswith("__labels"):
                continue
            var.set(float(self.gen_defaults.get(key, 0)))
            self._gen_value_changed(key)

    def _gen_new_seed(self):
        import random as _r
        self.gen_vars["seed"].set(float(_r.randint(0, 9999)))
        self._gen_value_changed("seed")

    # ---- rendering (off the main thread) --------------------------------
    def _gen_preview(self):
        if self.gen_busy:
            return
        self.gen_busy = True
        params = self._gen_params()
        self.gen_status.config(text="Generating...")
        threading.Thread(target=self._gen_worker, args=(params,), daemon=True).start()

    def _gen_worker(self, params):
        try:
            y, _specs = sgen.render(params)
            self.gen_q.put(("preview", y, params, sgen.describe(y)))
        except Exception:
            self.gen_q.put(("error", traceback.format_exc(limit=3), None, None))

    def _gen_series(self):
        if self.gen_busy:
            return
        dest = self.v_gen_dest.get().strip() or self.v_stimdir.get().strip()
        if not dest:
            messagebox.showwarning("No sound folder", "Choose where to save the set.")
            return
        try:
            key = self.v_gen_series_key.get()
            raw = self.v_gen_series_vals.get().replace(",", " ").split()
            values = [int(v) if key in sgen.INT_PARAMS else float(v) for v in raw]
        except ValueError:
            messagebox.showwarning("Check the values",
                                   "The steps must be numbers, separated by commas.")
            return
        if not values:
            return
        if not messagebox.askyesno(
                "Generate the set?",
                f"This makes {len(values)} stimuli, one per step of {key}, and saves them "
                f"into:\n{dest}\n\nContinue?"):
            return
        self.gen_busy = True
        self.gen_status.config(text=f"Generating {len(values)} stimuli...")
        params = self._gen_params()
        threading.Thread(target=self._gen_series_worker,
                         args=(params, key, values, dest), daemon=True).start()

    def _gen_series_worker(self, params, key, values, dest):
        try:
            outs = sgen.series(params, key, values, dest,
                               name_prefix=f"gen_{self.session_id}",
                               progress=lambda i, n, v: self.gen_q.put(
                                   ("progress", i, n, v)))
            self.gen_q.put(("series", outs, dest, key))
        except Exception:
            self.gen_q.put(("error", traceback.format_exc(limit=3), None, None))

    def _gen_poll(self):
        while True:
            try:
                msg = self.gen_q.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "preview":
                _, y, params, desc = msg
                self.gen_busy = False
                self.gen_audio = (y, params)
                self.gen_status.config(text="")
                self._gen_draw(y, params, desc)
                self._audio_play(y, sgen.SR, "the generated call")
            elif kind == "progress":
                _, i, n, v = msg
                self.gen_status.config(text=f"Generating {i}/{n}...")
            elif kind == "series":
                _, outs, dest, key = msg
                self.gen_busy = False
                self.gen_status.config(text=f"Saved {len(outs)} stimuli.")
                self.say(self.gen_box,
                         f"Made {len(outs)} stimuli varying {key}, saved into {dest}:\n" +
                         "\n".join("  " + os.path.basename(o) for o in outs) +
                         "\n\nThey are in your sound folder now, ready for Design Experiment.")
                self._log_event("generated_series", key=key, n=len(outs), folder=dest)
                if os.path.abspath(dest) == os.path.abspath(self.v_stimdir.get().strip() or "?"):
                    self.refresh_sounds()
            elif kind == "error":
                self.gen_busy = False
                self.gen_status.config(text="Problem.")
                self.say(self.gen_box, msg[1])
        self.root.after(200, self._gen_poll)

    def _gen_draw(self, y, params, desc):
        if self.gen_canvas is None:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            fig = Figure(figsize=(9, 2.4), dpi=90)
            self.gen_ax = fig.add_subplot(111)
            fig.tight_layout()
            self.gen_fig = fig
            self.gen_canvas = FigureCanvasTkAgg(fig, master=self.gen_view_holder)
            self.gen_canvas.get_tk_widget().pack(fill="both", expand=True)
        sp = self._sb_display_spec(y, sgen.SR)
        ax = self.gen_ax
        ax.clear()
        if len(sp["t"]) > 2:
            img = np.abs(sp["deriv"]).T
            ax.imshow(img, aspect="auto", origin="lower", cmap="Greys",
                      vmin=0, vmax=np.percentile(img, 99.5) or 1.0,
                      extent=[0, sp["t"][-1], sp["freqs"][0] / 1000, sp["freqs"][-1] / 1000])
        ax.set_ylabel("kHz", fontsize=8)
        ax.set_xlabel("seconds", fontsize=8)
        ax.tick_params(labelsize=7)
        self.gen_fig.tight_layout()
        self.gen_canvas.draw_idle()

        supern = sgen.supernormal_settings(params)
        lines = [f"{desc.get('seconds', 0):.2f} s, {desc.get('calls', 0)} call(s) detected."]
        if desc.get("calls"):
            lines.append(f"  measured: {desc['median_duration_ms']:.0f} ms each, "
                         f"centre {desc['median_centre_hz']:.0f} Hz, "
                         f"entropy {desc['median_entropy']:.2f}, FM {desc['median_fm']:.0f} deg")
            lines.append("  (measured with the same code as the Sound Library tab, so you can "
                         "compare these numbers directly against real recordings)")
        if supern:
            lines.append("  SUPERNORMAL settings: " + ", ".join(supern))
        self.say(self.gen_box, "\n".join(lines))

    def _gen_replay(self):
        if self.gen_audio is None:
            messagebox.showinfo("Nothing to play", "Press Listen first.")
            return
        self._audio_play(self.gen_audio[0], sgen.SR, "the generated call")

    def _gen_save(self):
        if self.gen_audio is None:
            messagebox.showinfo("Nothing to save", "Press Listen first.")
            return
        dest = self.v_gen_dest.get().strip() or self.v_stimdir.get().strip()
        if not dest:
            messagebox.showwarning("No sound folder", "Choose where to save it.")
            return
        y, params = self.gen_audio
        try:
            out = sgen.save_stimulus(y, dest, params, note=f"Sound Bench {self.session_id}")
        except Exception as e:
            messagebox.showerror("Could not save", str(e))
            return
        self._sb_append(self.gen_box, f"\nSaved {os.path.basename(out)} into {dest}\n")
        self._log_event("generated_stimulus", stimulus=out, parameters=params)
        if os.path.abspath(dest) == os.path.abspath(self.v_stimdir.get().strip() or "?"):
            self.refresh_sounds()
        messagebox.showinfo("Stimulus saved",
                            f"{os.path.basename(out)} is now in your sound folder.\n\n"
                            "It will appear on the Design Experiment tab.")


    # ==================================================== TAB: SOUND LIBRARY
    def _build_library(self, tab):
        ttk.Label(tab, text="Sound library",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6, 0))
        ttk.Label(tab, wraplength=820, justify="left", foreground="gray",
                  text="Look through your recordings, measure them, and cut the best calls "
                       "into stimuli. Anything you save here appears on the Design Experiment "
                       "tab as a sound you can play.").pack(anchor="w", pady=(2, 8))

        top = ttk.Frame(tab); top.pack(fill="x", pady=2)
        ttk.Label(top, text="Recordings folder:").pack(side="left")
        self.v_sb_libdir = tk.StringVar(value=os.path.join(HERE_SB, "song_data"))
        ttk.Entry(top, textvariable=self.v_sb_libdir).pack(side="left", fill="x",
                                                           expand=True, padx=6)
        ttk.Button(top, text="Browse...", command=self._sb_browse_libdir).pack(side="left")

        row = ttk.Frame(tab); row.pack(fill="x", pady=4)
        ttk.Button(row, text="Find recordings", command=self._sb_scan).pack(side="left")
        ttk.Button(row, text="Measure all", command=self._sb_measure_all).pack(side="left", padx=6)
        self.v_sb_rejected = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Include rejected recordings",
                        variable=self.v_sb_rejected).pack(side="left", padx=8)
        self.sb_lib_status = ttk.Label(row, text="", foreground="gray")
        self.sb_lib_status.pack(side="left", padx=8)

        cols = ("file", "bird", "day", "seconds", "calls")
        self.sb_tree = ttk.Treeview(tab, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (330, 80, 100, 70, 60)):
            self.sb_tree.heading(c, text=c.capitalize())
            self.sb_tree.column(c, width=w, anchor="w")
        self.sb_tree.pack(fill="x", pady=4)
        self.sb_tree.bind("<<TreeviewSelect>>", self._sb_select_file)
        self.sb_tree.bind("<Double-1>", lambda e: self._sb_play_file())   # double-click to hear it

        # ---- segmentation controls ----
        seg = ttk.LabelFrame(tab, text="What counts as a call")
        seg.pack(fill="x", pady=6)
        g = ttk.Frame(seg); g.pack(fill="x", padx=8, pady=6)
        self.v_sb_segfeat = tk.StringVar(value="amplitude")
        self.v_sb_thresh = tk.StringVar(value="25")
        self.v_sb_mindur = tk.StringVar(value="15")
        self.v_sb_mingap = tk.StringVar(value="8")
        self.v_sb_noise = tk.StringVar(value="1000")
        ttk.Label(g, text="Split on:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(g, textvariable=self.v_sb_segfeat, values=sfx.FEATURES, width=16,
                     state="readonly").grid(row=0, column=1, sticky="w", pady=3)
        self._grid_row(g, 1, "Threshold (dB below the loudest):", self.v_sb_thresh,
                       hint="lower = fewer, louder calls")
        self._grid_row(g, 2, "Shortest call (ms):", self.v_sb_mindur, hint="rejects clicks")
        self._grid_row(g, 3, "Merge calls closer than (ms):", self.v_sb_mingap)
        self._grid_row(g, 4, "Ignore sound below (Hz):", self.v_sb_noise,
                       hint="cage rumble; 0 turns this off")

        band = ttk.Frame(seg); band.pack(fill="x", padx=8, pady=(0, 8))
        self.v_sb_flow = tk.StringVar(value=str(int(sfx.SPEC_DEFAULTS["f_low"])))
        self.v_sb_fhigh = tk.StringVar(value=str(int(sfx.SPEC_DEFAULTS["f_high"])))
        ttk.Label(band, text="Measure between").pack(side="left")
        ttk.Entry(band, textvariable=self.v_sb_flow, width=7).pack(side="left", padx=4)
        ttk.Label(band, text="and").pack(side="left")
        ttk.Entry(band, textvariable=self.v_sb_fhigh, width=7).pack(side="left", padx=4)
        ttk.Label(band, text="Hz  (measured band; begging calls reach ~14 kHz)",
                  foreground="gray").pack(side="left")

        disp = ttk.Frame(seg); disp.pack(fill="x", padx=8, pady=(0, 8))
        self.v_sb_dlow = tk.StringVar(value=str(int(sfx.DISPLAY_DEFAULTS["f_low"])))
        self.v_sb_dhigh = tk.StringVar(value=str(int(sfx.DISPLAY_DEFAULTS["f_high"])))
        ttk.Label(disp, text="Show sonogram from").pack(side="left")
        ttk.Entry(disp, textvariable=self.v_sb_dlow, width=7).pack(side="left", padx=4)
        ttk.Label(disp, text="to").pack(side="left")
        ttk.Entry(disp, textvariable=self.v_sb_dhigh, width=7).pack(side="left", padx=4)
        ttk.Label(disp, text="Hz  (display only - does not change any measurement)",
                  foreground="gray").pack(side="left")
        ttk.Button(disp, text="Apply", command=self._sb_redraw_current).pack(side="left", padx=12)

        # ---- viewer ----
        self.v_sb_overlay = tk.StringVar(value="amplitude")
        vrow = ttk.Frame(tab); vrow.pack(fill="x", pady=(4, 0))
        ttk.Label(vrow, text="Show feature:").pack(side="left")
        ov = ttk.Combobox(vrow, textvariable=self.v_sb_overlay, values=sfx.FEATURES,
                          width=16, state="readonly")
        ov.pack(side="left", padx=4)
        ov.bind("<<ComboboxSelected>>", lambda e: self._sb_redraw_current())
        ttk.Button(vrow, text="< Earlier", command=lambda: self._sb_page(-1)).pack(side="left", padx=(16, 2))
        ttk.Button(vrow, text="Later >", command=lambda: self._sb_page(1)).pack(side="left")
        self.sb_view_label = ttk.Label(vrow, text="", foreground="gray")
        self.sb_view_label.pack(side="left", padx=10)

        self.sb_view_holder = ttk.Frame(tab, height=340)
        self.sb_view_holder.pack(fill="both", expand=True, pady=4)
        self.sb_view_canvas = None

        # ---- selection + export ----
        exp = ttk.LabelFrame(tab, text="Make a stimulus from the selection")
        exp.pack(fill="x", pady=6)
        srow = ttk.Frame(exp); srow.pack(fill="x", padx=8, pady=6)
        ttk.Label(srow, text="Drag across the sonogram to choose a stretch.  From").pack(side="left")
        self.v_sb_start = tk.StringVar(value="")
        self.v_sb_end = tk.StringVar(value="")
        ttk.Entry(srow, textvariable=self.v_sb_start, width=8).pack(side="left", padx=3)
        ttk.Label(srow, text="to").pack(side="left")
        ttk.Entry(srow, textvariable=self.v_sb_end, width=8).pack(side="left", padx=3)
        ttk.Label(srow, text="seconds").pack(side="left")
        ttk.Button(srow, text="Play selection",
                   command=self._sb_play_selection).pack(side="left", padx=(12, 4))
        ttk.Button(srow, text="Play whole file",
                   command=self._sb_play_file).pack(side="left", padx=2)
        ttk.Button(srow, text="Stop", command=self._audio_stop).pack(side="left", padx=2)
        ttk.Button(srow, text="Select whole file",
                   command=self._sb_select_all).pack(side="left", padx=(8, 0))

        erow = ttk.Frame(exp); erow.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(erow, text="Save into sound folder:").pack(side="left")
        self.v_sb_dest = tk.StringVar(value="")
        ttk.Entry(erow, textvariable=self.v_sb_dest, width=44).pack(side="left", padx=6)
        ttk.Button(erow, text="Use Design tab's folder",
                   command=lambda: self.v_sb_dest.set(self.v_stimdir.get())).pack(side="left")
        ttk.Button(erow, text="Use as stimulus",
                   command=self._sb_export).pack(side="left", padx=8)

        ttk.Button(tab, text="Save measurements as CSV...",
                   command=self._sb_export_table).pack(anchor="w", pady=(2, 0))
        self.sb_lib_box = tk.Text(tab, height=6, width=100, state="disabled")
        self.sb_lib_box.pack(fill="x", pady=4)

        self.root.after(200, self._sb_poll_library)

    # ---- scanning -------------------------------------------------------
    def _sb_browse_libdir(self):
        d = filedialog.askdirectory(title="Folder holding your recordings",
                                    initialdir=self.v_sb_libdir.get() or HERE_SB)
        if d:
            self.v_sb_libdir.set(d)
            self._sb_scan()

    def _sb_scan(self):
        folder = self.v_sb_libdir.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No folder", f"Cannot find the folder:\n{folder}")
            return
        try:
            t = slib.scan_folder(folder, include_rejected=self.v_sb_rejected.get())
        except Exception as e:
            messagebox.showerror("Could not read the folder", str(e))
            return
        self.sb_scan = t
        self.sb_tree.delete(*self.sb_tree.get_children())
        for _, r in t.iterrows():
            name = r["file"] + ("   [rejected]" if r["rejected"] else "")
            self.sb_tree.insert("", "end", values=(name, r["bird"], r["day"],
                                                   f"{r['duration_s']:.2f}", ""))
        self.sb_lib_status.config(text=f"{len(t)} recording(s).")
        self.say(self.sb_lib_box, f"Found {len(t)} recording(s) in {folder}.\n"
                                  "Click one to see it, or press 'Measure all'.")

    def _sb_measure_all(self):
        if self.sb_scan is None or self.sb_scan.empty:
            self._sb_scan()
        if self.sb_scan is None or self.sb_scan.empty:
            return
        paths = list(self.sb_scan["path"])
        self.sb_stop_scan.clear()
        self.sb_lib_status.config(text="Measuring...")
        threading.Thread(target=self._sb_measure_worker, args=(paths,), daemon=True).start()

    def _sb_measure_worker(self, paths):
        try:
            lib = self._sb_library()
            table = lib.analyze_many(
                paths, spec_params=self._sb_spec_params(),
                segment_params=self._sb_segment_params(),
                progress=lambda i, n, p: self.sb_libq.put(("progress", i, n, p)),
                stop=self.sb_stop_scan.is_set)
            self.sb_libq.put(("done", table))
        except Exception:
            self.sb_libq.put(("error", traceback.format_exc(limit=3)))

    def _sb_poll_library(self):
        while True:
            try:
                msg = self.sb_libq.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "progress":
                _, i, n, p = msg
                self.sb_lib_status.config(text=f"Measuring {i}/{n}...")
            elif msg[0] == "done":
                self.sb_table = msg[1]
                self._sb_fill_counts()
            elif msg[0] == "error":
                self.sb_lib_status.config(text="Problem while measuring.")
                self.say(self.sb_lib_box, msg[1])
        self.root.after(200, self._sb_poll_library)

    def _sb_fill_counts(self):
        table = self.sb_table
        self.sb_lib_status.config(text="Measured.")
        if table is None or table.empty:
            self.say(self.sb_lib_box, "No calls found with these settings. "
                                      "Try raising the threshold value.")
            return
        counts = table.groupby("file").size().to_dict() if "file" in table else {}
        for item in self.sb_tree.get_children():
            vals = list(self.sb_tree.item(item, "values"))
            name = vals[0].replace("   [rejected]", "")
            vals[4] = str(counts.get(name, 0))
            self.sb_tree.item(item, values=vals)
        lib = self._sb_library()
        s = lib.summary(table)
        total = int(s["syllables"].sum()) if len(s) else 0
        self.say(self.sb_lib_box,
                 f"{total} calls in {len(s)} file(s).\n" +
                 s.head(12).to_string(index=False))

    # ---- viewer ---------------------------------------------------------
    def _sb_select_file(self, _event=None):
        sel = self.sb_tree.selection()
        if not sel or self.sb_scan is None:
            return
        idx = self.sb_tree.index(sel[0])
        if idx >= len(self.sb_scan):
            return
        self.sb_path = self.sb_scan.iloc[idx]["path"]
        self.sb_page = 0.0
        self.sb_sel = None
        self.v_sb_start.set(""); self.v_sb_end.set("")
        self._sb_load_and_draw()

    def _sb_redraw_current(self):
        if self.sb_path:
            self._sb_load_and_draw()

    def _sb_page(self, direction):
        if self.sb_audio is None:
            return
        total = len(self.sb_audio) / self.sb_fs
        self.sb_page = max(0.0, min(max(0.0, total - PAGE_SECONDS),
                                    self.sb_page + direction * PAGE_SECONDS * 0.8))
        self._sb_load_and_draw(reuse=True)

    def _sb_load_and_draw(self, reuse=False):
        try:
            if not reuse or self.sb_audio is None:
                self.sb_audio, self.sb_fs = slib.read_audio(self.sb_path)
            a = int(self.sb_page * self.sb_fs)
            b = min(len(self.sb_audio), a + int(PAGE_SECONDS * self.sb_fs))
            x = self.sb_audio[a:b]
            feat, syl, sp = sfx.analyze(x, self.sb_fs,
                                        spec_params=self._sb_spec_params(),
                                        segment_params=self._sb_segment_params())
            self.sb_feat, self.sb_syl, self.sb_spec = feat, syl, sp
            self.sb_disp = self._sb_display_spec(x, self.sb_fs)
        except Exception as e:
            messagebox.showerror("Could not measure this recording", str(e))
            return
        total = len(self.sb_audio) / self.sb_fs
        self.sb_view_label.config(
            text=f"{os.path.basename(self.sb_path)}   "
                 f"{self.sb_page:.1f}-{min(total, self.sb_page + PAGE_SECONDS):.1f} s "
                 f"of {total:.1f} s   {len(syl)} call(s) here")
        self._sb_draw_view()

    def _sb_view_axes(self):
        if self.sb_view_canvas is None:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            fig = Figure(figsize=(9, 3.6), dpi=90)
            self.sb_ax_spec = fig.add_subplot(211)
            self.sb_ax_feat = fig.add_subplot(212, sharex=self.sb_ax_spec)
            fig.tight_layout()
            self.sb_view_fig = fig
            self.sb_view_canvas = FigureCanvasTkAgg(fig, master=self.sb_view_holder)
            self.sb_view_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.sb_view_canvas.mpl_connect("button_press_event", self._sb_press)
            self.sb_view_canvas.mpl_connect("button_release_event", self._sb_release)
        return self.sb_ax_spec, self.sb_ax_feat

    def _sb_draw_view(self):
        ax1, ax2 = self._sb_view_axes()
        sp, feat, syl = getattr(self, "sb_disp", None) or self.sb_spec, self.sb_feat, self.sb_syl
        off = self.sb_page
        ax1.clear(); ax2.clear()
        if len(sp["t"]) > 2:
            img = np.abs(sp["deriv"]).T
            vmax = np.percentile(img, 99.5) or 1.0
            ax1.imshow(img, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=vmax,
                       extent=[off, off + sp["t"][-1], sp["freqs"][0] / 1000,
                               sp["freqs"][-1] / 1000])
        ax1.set_ylabel("kHz", fontsize=8)
        ax1.tick_params(labelsize=7)

        col = self.v_sb_overlay.get()
        if col in feat:
            ax2.plot(feat["t"].to_numpy() + off, feat[col].to_numpy(), lw=0.8, color="#1d4ed8")
        ax2.set_ylabel(sfx.FEATURE_LABELS.get(col, col), fontsize=8)
        ax2.set_xlabel("seconds", fontsize=8)
        ax2.tick_params(labelsize=7)

        for _, r in syl.iterrows():          # call boundaries, on both panels
            for ax in (ax1, ax2):
                ax.axvspan(r["start_s"] + off, r["end_s"] + off, color="#f59e0b", alpha=0.18)
        if self.sb_sel:
            for ax in (ax1, ax2):
                ax.axvspan(self.sb_sel[0], self.sb_sel[1], color="#16a34a", alpha=0.25)
        self.sb_view_fig.tight_layout()
        self.sb_view_canvas.draw_idle()

    def _sb_press(self, event):
        if event.xdata is not None:
            self._sb_drag_from = float(event.xdata)

    def _sb_release(self, event):
        if event.xdata is None or getattr(self, "_sb_drag_from", None) is None:
            return
        a, b = sorted([self._sb_drag_from, float(event.xdata)])
        if b - a < 0.005:
            return
        self.sb_sel = (a, b)
        self.v_sb_start.set(f"{a:.3f}")
        self.v_sb_end.set(f"{b:.3f}")
        self._sb_draw_view()

    def _sb_select_all(self):
        if self.sb_audio is None:
            return
        self.sb_sel = (0.0, len(self.sb_audio) / self.sb_fs)
        self.v_sb_start.set("0.000")
        self.v_sb_end.set(f"{self.sb_sel[1]:.3f}")
        self._sb_draw_view()

    def _sb_current_selection(self):
        try:
            a = float(self.v_sb_start.get()); b = float(self.v_sb_end.get())
        except ValueError:
            return None
        return (a, b) if b > a else None

    def _sb_play_file(self):
        """Hear the whole recording, without having to select a stretch first."""
        if self.sb_audio is None:
            messagebox.showinfo("Nothing chosen", "Pick a recording from the list first.")
            return
        self._audio_play(self.sb_audio, self.sb_fs, os.path.basename(self.sb_path or ""))

    def _sb_play_selection(self):
        sel = self._sb_current_selection()
        if self.sb_audio is None or sel is None:
            messagebox.showinfo("Nothing selected",
                                "Choose a recording, then drag across the sonogram.")
            return
        a, b = int(sel[0] * self.sb_fs), int(sel[1] * self.sb_fs)
        self._audio_play(self.sb_audio[a:b], self.sb_fs,
                         f"{os.path.basename(self.sb_path or '')} {sel[0]:.2f}-{sel[1]:.2f}s")

    # ---- export ---------------------------------------------------------
    def _sb_export(self):
        sel = self._sb_current_selection()
        if not self.sb_path or sel is None:
            messagebox.showinfo("Nothing to save",
                                "Choose a recording and drag across the sonogram first.")
            return
        dest = self.v_sb_dest.get().strip() or self.v_stimdir.get().strip()
        if not dest:
            messagebox.showwarning("No sound folder",
                                   "Choose the folder your stimuli live in.")
            return
        try:
            out = slib.export_stimulus(self.sb_path, dest, start_s=sel[0], end_s=sel[1],
                                       note=f"Sound Bench {self.session_id}")
        except Exception as e:
            messagebox.showerror("Could not save the stimulus", str(e))
            return
        self._sb_append(self.sb_lib_box,
                        f"Saved {os.path.basename(out)}  ({sel[1] - sel[0]:.2f} s) into {dest}\n")
        self._log_event("stimulus_exported", source=self.sb_path,
                        start_s=sel[0], end_s=sel[1], stimulus=out)
        # Design Experiment reads that folder, so refresh it now.
        if os.path.abspath(dest) == os.path.abspath(self.v_stimdir.get().strip() or "?"):
            self.refresh_sounds()
        messagebox.showinfo("Stimulus saved",
                            f"{os.path.basename(out)} is now in your sound folder.\n\n"
                            "It will appear on the Design Experiment tab.")

    def _sb_export_table(self):
        if self.sb_table is None or self.sb_table.empty:
            messagebox.showinfo("Nothing measured yet", "Press 'Measure all' first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save measurements", defaultextension=".csv",
            initialfile="call_measurements.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.sb_table.to_csv(path, index=False)
        self._sb_append(self.sb_lib_box, f"Saved measurements to {path}\n")


HERE_SB = os.path.dirname(os.path.abspath(__file__))
