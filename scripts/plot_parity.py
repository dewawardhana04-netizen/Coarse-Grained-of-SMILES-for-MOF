"""
plot_parity.py
Grafik parity (prediksi vs sebenarnya) + GARIS REGRESI LINEAR beserta
statistik kecocokannya.

Dua garis ditampilkan dan artinya BERBEDA:
  - Garis putus-putus abu  : y = x (parity ideal). Bukan hasil fit.
  - Garis penuh merah      : regresi linear y_pred = a * y_asli + b.
        a  -> kemiringan. a < 1 berarti model meratakan prediksi
              (underestimasi nilai tinggi, overestimasi nilai rendah).
        b  -> pergeseran sistematis.
        r  -> korelasi Pearson.  R2 -> koefisien determinasi model.

Pakai:
    python3 plot_parity.py
    python3 plot_parity.py prediksi_test.csv Test
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def statistik(x, y):
    """Regresi linear kuadrat terkecil + metrik terkait."""
    a, b = np.polyfit(x, y, 1)                 # y = a*x + b
    r = float(np.corrcoef(x, y)[0, 1])
    ss_res = float(((y - x) ** 2).sum())       # residual terhadap y = x
    ss_tot = float(((x - x.mean()) ** 2).sum())
    r2_model = 1 - ss_res / ss_tot             # R2 model (bukan R2 garis fit)
    rmse = float(np.sqrt(((y - x) ** 2).mean()))
    mae = float(np.abs(y - x).mean())
    return {"slope": float(a), "intercept": float(b), "pearson_r": r,
            "r2_fit": r ** 2, "r2_model": r2_model, "rmse": rmse, "mae": mae}


def main(berkas="prediksi_test.csv", label="Test"):
    df = pd.read_csv(berkas)
    kol_x = "y_asli" if "y_asli" in df else df.columns[-2]
    kol_y = "y_prediksi" if "y_prediksi" in df else df.columns[-1]
    x = df[kol_x].to_numpy(dtype=float)
    y = df[kol_y].to_numpy(dtype=float)

    s = statistik(x, y)

    print("=" * 58)
    print(f"REGRESI LINEAR — {label}  (n = {len(x):,})")
    print("=" * 58)
    print(f"Persamaan garis : y_pred = {s['slope']:.4f} * y_asli "
          f"{'+' if s['intercept'] >= 0 else '-'} {abs(s['intercept']):.4f}")
    print(f"Slope (a)       : {s['slope']:.4f}   (ideal = 1)")
    print(f"Intercept (b)   : {s['intercept']:.4f}   (ideal = 0)")
    print(f"Pearson r       : {s['pearson_r']:.4f}")
    print(f"R2 garis fit    : {s['r2_fit']:.4f}   <- seberapa linear hubungannya")
    print(f"R2 model        : {s['r2_model']:.4f}   <- akurasi terhadap y = x")
    print(f"RMSE            : {s['rmse']:.4f}")
    print(f"MAE             : {s['mae']:.4f}")
    print("-" * 58)
    if s["slope"] < 0.9:
        print("Catatan: slope < 0,9 -> model MERATAKAN prediksi; nilai tinggi")
        print("         cenderung diprediksi terlalu rendah (regression to mean).")
    elif s["slope"] > 1.1:
        print("Catatan: slope > 1,1 -> model MELEBIH-LEBIHKAN rentang prediksi.")
    else:
        print("Catatan: slope mendekati 1 -> tidak ada bias skala yang berarti.")

    # ---------------- grafik ----------------
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
    ax.scatter(x, y, s=7, alpha=0.28, color="#2b4a75", edgecolors="none",
               label=f"data ({len(x):,} titik)")

    lo = 0.0
    hi = max(x.max(), y.max()) * 1.05
    grid = np.linspace(lo, hi, 100)

    ax.plot(grid, grid, "--", color="#8a93a4", lw=1.3, label="parity ideal (y = x)")
    ax.plot(grid, s["slope"] * grid + s["intercept"], "-", color="#c0392b",
            lw=1.8,
            label=f"regresi: y = {s['slope']:.3f}x "
                  f"{'+' if s['intercept'] >= 0 else '-'} {abs(s['intercept']):.3f}")

    kotak = (f"$R^2_{{model}}$ = {s['r2_model']:.3f}\n"
             f"$r$ = {s['pearson_r']:.3f}\n"
             f"RMSE = {s['rmse']:.3f}\n"
             f"MAE = {s['mae']:.3f}")
    ax.text(0.04, 0.96, kotak, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c8cedb", lw=0.6))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Uptake CO$_2$ sebenarnya (mol/kg)", fontsize=10.5)
    ax.set_ylabel("Uptake CO$_2$ prediksi (mol/kg)", fontsize=10.5)
    ax.set_title(f"Parity plot MOFHAPPY — {label}", fontsize=12.5,
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    nama = f"parity_{label.lower()}.png"
    fig.savefig(nama)
    print(f"\nGrafik disimpan : {nama}")

    # ---------------- residual ----------------
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    res = y - x
    ax2.scatter(x, res, s=7, alpha=0.28, color="#2b4a75", edgecolors="none")
    ax2.axhline(0, color="#c0392b", ls="--", lw=1.3)
    ax2.set_xlabel("Uptake CO$_2$ sebenarnya (mol/kg)", fontsize=10.5)
    ax2.set_ylabel("Residual (prediksi $-$ asli)", fontsize=10.5)
    ax2.set_title(f"Plot residual — {label}", fontsize=12, fontweight="bold",
                  pad=10)
    ax2.grid(alpha=0.22, lw=0.6); ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig2.tight_layout()
    nama2 = f"residual_{label.lower()}.png"
    fig2.savefig(nama2)
    print(f"Grafik disimpan : {nama2}")

    pd.DataFrame([s]).to_csv(f"statistik_regresi_{label.lower()}.csv",
                             index=False)
    print(f"Statistik       : statistik_regresi_{label.lower()}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prediksi_test.csv",
         sys.argv[2] if len(sys.argv) > 2 else "Test")
