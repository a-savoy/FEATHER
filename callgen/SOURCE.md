# Vendored from the Hugging Face Space

These files are copied **unmodified** from
<https://huggingface.co/spaces/a-savoy/zf-begging-call-generator>

- revision: `aac7233f51e39cb0586a712ec5457bda94a393ef`
- fetched: 2026-09-02

`generator_engine.py`, `audio_utils.py` and `train_vae.py` are the Space's own
modules; the `.pt` / `.npz` / `.json` files are the trained model and its
normalisation, latent directions, and empirical parameter ranges.

Keeping them byte-identical means the Space stays the upstream: to update, re-download
the same file list and replace this folder. FEATHER never edits them — everything
FEATHER-specific lives in `song_generate.py` one level up.

The Gradio front end (`app.py`, `call_gui_web.py`) is deliberately NOT vendored;
FEATHER provides its own Tk interface over the same engine.

## Note on `plots/`

`train_vae.py` creates an empty `plots/` folder when it is imported (it is the training
script, and FEATHER only imports it for the model class and its constants). Harmless,
and left alone so these files stay byte-identical to the Space.
