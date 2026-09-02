"""
Version 2 conditional VAE on log-mel call spectrograms — POOLED across birds.

Conditioning vector (3-d, was 11-d in the bird-ID version):
    [age_n, duration_n, sex]
  where sex = 0 for males, 1 for females. All 9 birds pooled into one dataset;
  bird identity is not a conditioning input.

Rationale: the corpus's bird-level variance was absorbing latent capacity that
would otherwise encode within-condition acoustic variation. Sex and age are the
ethologically meaningful axes; bird identity was a nuisance variable.

Otherwise identical to the v1 recipe: 16-d latent, L1+MSE hybrid loss, 30
epochs on CPU (~20 min), file-level train/val split.
"""

from __future__ import annotations
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
CKPT = HERE / "vae_ckpt.pt"
NORM = HERE / "vae_norm_stats.npz"
LOG_CSV = HERE / "vae_training_log.csv"
PLOTS.mkdir(exist_ok=True)

NPZ = HERE / "melspec_dataset.npz"
CSV = HERE / "melspec_metadata.csv"

DAY_MIN, DAY_MAX = 4, 15
DUR_MIN, DUR_MAX = 0.04, 0.40        # seconds

LATENT = 16
COND_DIM = 3                          # age + duration + sex
BATCH = 128
EPOCHS = 30
LR = 3e-4
BETA_MAX = 0.25
BETA_WARMUP_EPOCHS = 10
LOGVAR_CLAMP = 5.0
GRAD_CLIP = 1.0
SEED = 0
TIME_PAD_TO = 64
FLOOR = -23.0259
VAL_FRAC = 0.15

L1_WEIGHT = 1.0
MSE_WEIGHT = 0.5


# ----------------------------- model -----------------------------

class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent=LATENT, cond_dim=COND_DIM):
        super().__init__()
        self.fc = nn.Linear(latent + cond_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z, cond):
        h = self.fc(torch.cat([z, cond], dim=1))
        h = h.view(-1, 256, 4, 4)
        return self.net(h)


class CVAE(nn.Module):
    def __init__(self, latent=LATENT, cond_dim=COND_DIM):
        super().__init__()
        self.latent = latent
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent, cond_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, cond)
        return x_hat, mu, logvar


# ----------------------------- data -----------------------------

def pad_time(x: np.ndarray, target: int = TIME_PAD_TO, floor: float = FLOOR) -> np.ndarray:
    T = x.shape[-1]
    if T == target: return x
    if T > target: return x[..., :target]
    pad = target - T
    pad_shape = x.shape[:-1] + (pad,)
    return np.concatenate([x, np.full(pad_shape, floor, dtype=x.dtype)], axis=-1)


class MelDataset(Dataset):
    """Pooled across birds. Conditions only on age, duration, sex."""
    def __init__(self, ms: np.ndarray, meta: pd.DataFrame, mean: float, std: float):
        self.ms = ms
        self.days = meta["day"].to_numpy(dtype=np.float32)
        self.durs = meta["duration_s"].to_numpy(dtype=np.float32)
        self.sexes = (meta["sex"].to_numpy() == "F").astype(np.float32)  # 0=M, 1=F
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.ms)

    def __getitem__(self, i):
        x = (self.ms[i] - self.mean) / self.std
        x = torch.from_numpy(x).unsqueeze(0)
        age_n = (self.days[i] - DAY_MIN) / (DAY_MAX - DAY_MIN)
        dur_n = float(np.clip((self.durs[i] - DUR_MIN) / (DUR_MAX - DUR_MIN), 0.0, 1.0))
        sex_n = float(self.sexes[i])
        cond = torch.from_numpy(np.array([age_n, dur_n, sex_n], dtype=np.float32))
        return x, cond


# ----------------------------- train -----------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    force_cpu = os.environ.get("FORCE_CPU", "1") == "1"
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    print("[load] spectrogram dataset")
    d = np.load(NPZ)
    ms = d["melspecs"].astype(np.float32)
    ms = pad_time(ms, TIME_PAD_TO, FLOOR)
    meta = pd.read_csv(CSV)
    assert len(meta) == len(ms)
    n_f = (meta["sex"] == "F").sum()
    n_m = (meta["sex"] == "M").sum()
    print(f"[load] ms {ms.shape}, meta {len(meta)}  (F={n_f}, M={n_m})")

    files = meta["file"].to_numpy()
    uniq_files = np.array(sorted(set(files.tolist())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq_files)
    n_val = int(VAL_FRAC * len(uniq_files))
    val_files = set(uniq_files[:n_val])
    is_val = np.array([f in val_files for f in files])
    print(f"[split] {(~is_val).sum()} train / {is_val.sum()} val "
          f"({len(uniq_files)-n_val}/{n_val} files)")

    tr_mean = float(ms[~is_val].mean())
    tr_std = float(ms[~is_val].std())
    print(f"[norm] mean={tr_mean:.3f}, std={tr_std:.3f}")

    ds_tr = MelDataset(ms[~is_val], meta[~is_val].reset_index(drop=True), tr_mean, tr_std)
    ds_va = MelDataset(ms[is_val], meta[is_val].reset_index(drop=True), tr_mean, tr_std)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)

    model = CVAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CVAE params={n_params/1e6:.2f}M  (cond_dim={COND_DIM})")

    def beta_at(epoch):
        if epoch >= BETA_WARMUP_EPOCHS: return BETA_MAX
        return BETA_MAX * (epoch + 1) / BETA_WARMUP_EPOCHS

    log_rows = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        beta = beta_at(epoch)
        model.train()
        n, tot_recon, tot_kl, n_skip = 0, 0.0, 0.0, 0
        for x, cond in dl_tr:
            x = x.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x, cond)
            l1 = (x_hat - x).abs().mean()
            mse = ((x_hat - x) ** 2).mean()
            recon = L1_WEIGHT * l1 + MSE_WEIGHT * mse
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
            loss = recon + beta * kl / (x.shape[-1] * x.shape[-2])
            if not torch.isfinite(loss):
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot_recon += recon.item() * x.size(0)
            tot_kl += kl.item() * x.size(0)
            n += x.size(0)
        tr_recon = tot_recon / n
        tr_kl = tot_kl / n

        model.eval()
        n, vr, vk = 0, 0.0, 0.0
        with torch.no_grad():
            for x, cond in dl_va:
                x = x.to(device); cond = cond.to(device)
                x_hat, mu, logvar = model(x, cond)
                l1 = (x_hat - x).abs().mean()
                mse = ((x_hat - x) ** 2).mean()
                recon = L1_WEIGHT * l1 + MSE_WEIGHT * mse
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
                vr += recon.item() * x.size(0)
                vk += kl.item() * x.size(0)
                n += x.size(0)
        va_recon = vr / n
        va_kl = vk / n
        dt = time.time() - t0
        print(f"[e{epoch:02d}] beta={beta:.2f} "
              f"tr_recon={tr_recon:.4f} tr_kl={tr_kl:.2f} "
              f"va_recon={va_recon:.4f} va_kl={va_kl:.2f} "
              f"skip={n_skip} ({dt:.1f}s elapsed)")
        log_rows.append(dict(epoch=epoch, beta=beta,
                             tr_recon=tr_recon, tr_kl=tr_kl,
                             va_recon=va_recon, va_kl=va_kl,
                             elapsed_s=dt))

    torch.save({
        "model": model.state_dict(),
        "latent": LATENT,
        "cond_dim": COND_DIM,
        "cond_layout": ["age", "duration", "sex"],
        "day_min": DAY_MIN, "day_max": DAY_MAX,
        "dur_min": DUR_MIN, "dur_max": DUR_MAX,
        "time_pad_to": TIME_PAD_TO,
        "l1_weight": L1_WEIGHT, "mse_weight": MSE_WEIGHT,
    }, CKPT)
    np.savez(NORM, mean=tr_mean, std=tr_std, floor=FLOOR)
    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)
    print(f"[save] {CKPT} / {NORM} / {LOG_CSV}")

    df = pd.DataFrame(log_rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df.epoch, df.tr_recon, label="train")
    ax[0].plot(df.epoch, df.va_recon, label="val")
    ax[0].set_title("Recon L1+MSE"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(df.epoch, df.tr_kl, label="train")
    ax[1].plot(df.epoch, df.va_kl, label="val")
    ax[1].set_title("KL (nats/sample)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(PLOTS / "vae_training_curves.png", dpi=130)
    plt.close(fig)
    print(f"[save] {PLOTS}/vae_training_curves.png")


if __name__ == "__main__":
    main()
