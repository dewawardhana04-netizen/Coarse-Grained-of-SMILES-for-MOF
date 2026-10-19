"""
banding_model.py
Visualisasi pembanding dari rekap_semua.csv + seluruh prediksi_*.csv.

Menghasilkan:
  1. banding_bar.png       - bar chart R2 / RMSE / MAE / slope antar-model
  2. banding_parity.png    - grid parity plot, satu panel per model
  3. banding_residual.png  - histogram residual bertumpuk
  4. banding_ablasi.png    - single vs attention per encoder (bukti multi-linker)
  5. banding_ringkas.csv   - tabel metrik yang dihitung ulang dari prediksi

Pakai:
    python3 banding_model.py
    python3 banding_model.py hasil            # folder lain
    python3 banding_model.py hasil linkerset  # hanya split tertentu
    python3 banding_model.py hasil all ann,rnn,lstm,gru   # sertakan GRU

Secara baku hanya ANN, RNN, dan LSTM yang ditampilkan (GRU dikecualikan).
"""
import sys, re, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WARNA = {"ann": "#c0392b", "rnn": "#e08c1a", "lstm": "#2b4a75"}

# Nama kondisi untuk grafik (urutan = urutan tampil)
KONDISI = {
    "concat_unshared": "1. Bobot belum sama",
    "concat_shared":   "2. Bobot disamakan",
    "minmaxavg":       "3. Min-Max-Avg",
}
URUT_ENC = ["ann", "rnn", "lstm"]

# Encoder yang DITAMPILKAN pada semua grafik. GRU dikecualikan secara
# baku; ubah lewat argumen ke-3 di CLI, mis. "ann,rnn,lstm,gru".
ENCODER_DIPAKAI = ["ann", "rnn", "lstm"]


def metrik(x, y):
    res = y - x
    mse = float((res ** 2).mean())
    a, b = np.polyfit(x, y, 1)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": float(np.abs(res).mean()),
            "R2": float(1 - (res ** 2).sum() / ((x - x.mean()) ** 2).sum()),
            "slope": float(a), "intercept": float(b),
            "r": float(np.corrcoef(x, y)[0, 1]), "n": len(x)}


def muat(dirh: Path, filter_split=None, encoder_dipakai=None):
    """Baca semua prediksi_*.csv dan uraikan namanya jadi kolom."""
    baris = []
    for f in sorted(dirh.glob("prediksi_*.csv")):
        # Nama berkas: prediksi_[repr_]<encoder>_<agregasi>_<split>.csv
        # agregasi bisa mengandung underscore (concat_unshared), jadi
        # diuraikan dari dua ujung, bukan dengan split biasa.
        bag = f.stem.replace("prediksi_", "").split("_")
        if len(bag) < 3:
            continue
        sp = bag[-1]
        REPR = {"mofhappy", "smiles"}
        ENC = {"ann", "rnn", "lstm", "gru"}
        rep = bag[0] if bag[0] in REPR else "mofhappy"
        sisa = bag[1:] if bag[0] in REPR else bag
        if sisa[0] not in ENC:
            continue
        enc = sisa[0]
        agg = "_".join(sisa[1:-1])
        if not agg:
            continue
        if filter_split and filter_split != "all" and sp != filter_split:
            continue
        if encoder_dipakai and enc not in encoder_dipakai:
            continue
        d = pd.read_csv(f)
        if not {"y_asli", "y_prediksi"}.issubset(d.columns):
            continue
        m = metrik(d["y_asli"].to_numpy(float), d["y_prediksi"].to_numpy(float))
        baris.append({"berkas": f.name, "representasi": rep, "encoder": enc,
                      "agregasi": agg, "split": sp,
                      "label": f"{enc.upper()}-{agg}", "data": d, **m})
    if not baris:
        raise SystemExit(f"Tidak ada prediksi_*.csv di {dirh}")
    return baris


def bar_chart(B, out):
    tab = pd.DataFrame([{k: v for k, v in b.items() if k != "data"}
                        for b in B]).sort_values("R2", ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    for ax, kol, judul, baik in zip(
            axes.ravel(), ["R2", "RMSE", "MAE", "slope"],
            ["$R^2$ (makin tinggi makin baik)", "RMSE (makin rendah makin baik)",
             "MAE (makin rendah makin baik)", "Slope regresi (ideal = 1)"],
            [True, False, False, None]):
        t = tab.sort_values(kol, ascending=not baik if baik is not None else False)
        warna = [WARNA.get(e, "#666") for e in t["encoder"]]
        ax.barh(t["label"], t[kol], color=warna, edgecolor="white", height=0.72)
        for i, v in enumerate(t[kol]):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=7.5)
        if kol == "slope":
            ax.axvline(1.0, ls="--", color="#c0392b", lw=1.2)
        ax.set_title(judul, fontsize=10.5, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Perbandingan Model MOFHAPPY", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    return tab


def grid_parity(B, out):
    B = sorted(B, key=lambda b: -b["R2"])
    n = len(B)
    kol = min(4, n)
    bar = int(np.ceil(n / kol))
    fig, axes = plt.subplots(bar, kol, figsize=(3.3 * kol, 3.4 * bar), dpi=140,
                             squeeze=False)
    hi = max(max(b["data"]["y_asli"].max(), b["data"]["y_prediksi"].max())
             for b in B) * 1.05
    for ax, b in zip(axes.ravel(), B):
        x = b["data"]["y_asli"].to_numpy(float)
        y = b["data"]["y_prediksi"].to_numpy(float)
        ax.scatter(x, y, s=3, alpha=0.18,
                   color=WARNA.get(b["encoder"], "#2b4a75"), edgecolors="none")
        g = np.linspace(0, hi, 50)
        ax.plot(g, g, "--", color="#8a93a4", lw=1.0)
        ax.plot(g, b["slope"] * g + b["intercept"], "-", color="#c0392b", lw=1.4)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_title(f"{b['label']}\n$R^2$={b['R2']:.3f}  slope={b['slope']:.3f}",
                     fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(B):]:
        ax.axis("off")
    fig.supxlabel("Uptake CO$_2$ sebenarnya (mol/kg)", fontsize=10)
    fig.supylabel("Uptake CO$_2$ prediksi (mol/kg)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def hist_residual(B, out):
    B = sorted(B, key=lambda b: -b["R2"])[:6]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    bins = np.linspace(-6, 6, 80)
    for b in B:
        r = b["data"]["y_prediksi"] - b["data"]["y_asli"]
        a1.hist(r, bins=bins, histtype="step", lw=1.5,
                color=WARNA.get(b["encoder"], "#666"),
                label=f"{b['label']}  MAE={b['MAE']:.3f}")
    a1.axvline(0, color="#333", ls="--", lw=1.0)
    a1.set_xlabel("Residual (prediksi $-$ asli), mol/kg")
    a1.set_ylabel("Jumlah titik")
    a1.set_title("Distribusi residual", fontsize=11, fontweight="bold")
    a1.legend(fontsize=7.5, frameon=False)

    # distribusi target vs prediksi model terbaik
    b = B[0]
    bins2 = np.linspace(0, float(b["data"]["y_asli"].max()), 60)
    a2.hist(b["data"]["y_asli"], bins=bins2, alpha=0.55, color="#8a93a4",
            label="nilai sebenarnya")
    a2.hist(b["data"]["y_prediksi"], bins=bins2, alpha=0.55,
            color=WARNA.get(b["encoder"], "#2b4a75"),
            label=f"prediksi {b['label']}")
    a2.set_xlabel("Uptake CO$_2$ (mol/kg)")
    a2.set_ylabel("Jumlah titik")
    a2.set_title("Sebaran target vs prediksi (model terbaik)",
                 fontsize=11, fontweight="bold")
    a2.legend(fontsize=8, frameon=False)
    for ax in (a1, a2):
        ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_kondisi(tab, out):
    """Bar chart 3 kondisi x 3 encoder: bukti perbaikan bertahap."""
    t = tab[tab["agregasi"].isin(KONDISI)]
    if t.empty:
        print("  (lewati banding_kondisi.png: mode kondisi belum ada)")
        return
    enc = [e for e in URUT_ENC if e in set(t["encoder"])]  # sudah tersaring
    kon = [k for k in KONDISI if k in set(t["agregasi"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)
    for ax, kol, judul, ideal in zip(
            axes, ["R2", "RMSE", "MAE"],
            ["$R^2$ (makin tinggi makin baik)", "RMSE (makin rendah makin baik)",
             "MAE (makin rendah makin baik)"], [None, None, None]):
        x = np.arange(len(enc)); w = 0.8 / max(len(kon), 1)
        for j, k in enumerate(kon):
            nilai = [t[(t.encoder == e) & (t.agregasi == k)][kol].mean()
                     for e in enc]
            ax.bar(x + (j - (len(kon) - 1) / 2) * w, nilai, w,
                   label=KONDISI[k],
                   color=["#b9c2d2", "#6b87ad", "#10233f"][j % 3],
                   edgecolor="white")
            for xi, v in zip(x + (j - (len(kon) - 1) / 2) * w, nilai):
                if not np.isnan(v):
                    ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([e.upper() for e in enc])
        ax.set_title(judul, fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Perbandingan tiga kondisi agregasi linker",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main(dirh="hasil", filter_split=None, encoder=None):
    dirh = Path(dirh)
    enc_pakai = ([e.strip().lower() for e in encoder.split(",")]
                 if encoder else ENCODER_DIPAKAI)
    B = muat(dirh, filter_split, enc_pakai)
    print(f"Encoder ditampilkan : {', '.join(e.upper() for e in enc_pakai)}")
    print(f"Ditemukan {len(B)} berkas prediksi di {dirh}")

    tab = bar_chart(B, "banding_bar.png")
    grid_parity(B, "banding_parity.png")
    hist_residual(B, "banding_residual.png")
    plot_kondisi(tab, "banding_kondisi.png")

    kol = ["encoder", "agregasi", "split", "n", "R2", "RMSE", "MAE",
           "slope", "r"]
    print("\n" + tab[kol].to_string(index=False))
    tab[kol].to_csv("banding_ringkas.csv", index=False)
    print("\nDisimpan: banding_bar.png, banding_parity.png, "
          "banding_residual.png, banding_kondisi.png, banding_ringkas.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hasil",
         sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
