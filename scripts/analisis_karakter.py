"""
analisis_karakter.py
Analisis distribusi jumlah karakter pada kolom `mofhappy`.

Pakai:  python3 analisis_karakter.py hasil_mofhappy.csv

Keluaran:
  - hasil_mofhappy_analisis.csv   (data + kolom jumlah_karakter)
  - frekuensi_panjang.csv         (tabel frekuensi tiap panjang)
  - histogram_karakter.png        (grafik distribusi)
  - ringkasan statistik di layar
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # aman dijalankan tanpa GUI
import matplotlib.pyplot as plt


def main(berkas="hasil_mofhappy.csv"):
    # --- 1. Baca dataset -----------------------------------------------
    df = pd.read_csv(berkas)
    if "mofhappy" not in df.columns:
        raise SystemExit(f"Kolom 'mofhappy' tidak ada. Kolom tersedia: "
                         f"{list(df.columns)}")

    n_awal = len(df)

    # --- 2-4. Hitung panjang karakter ----------------------------------
    # NaN ditangani lebih dulu: baris kosong TIDAK dihitung sebagai 0,
    # melainkan dikeluarkan, supaya tidak membuat puncak palsu di x=0.
    n_kosong = df["mofhappy"].isna().sum()
    df["jumlah_karakter"] = df["mofhappy"].astype("string").str.len()
    df_valid = df.dropna(subset=["jumlah_karakter"]).copy()
    df_valid["jumlah_karakter"] = df_valid["jumlah_karakter"].astype(int)

    p = df_valid["jumlah_karakter"]

    # --- 6-7. Statistik ------------------------------------------------
    frek = p.value_counts().sort_values(ascending=False)
    dominan = int(frek.index[0])
    jumlah_dominan = int(frek.iloc[0])

    print("=" * 58)
    print("RINGKASAN DISTRIBUSI PANJANG KARAKTER")
    print("=" * 58)
    print(f"Baris dibaca              : {n_awal:,}")
    print(f"Baris kosong/NaN (dibuang): {n_kosong:,}")
    print(f"Baris dianalisis          : {len(p):,}")
    print("-" * 58)
    print(f"Panjang paling dominan    : {dominan} karakter")
    print(f"Jumlah MOF pada panjang itu: {jumlah_dominan:,} "
          f"({100*jumlah_dominan/len(p):.1f}%)")
    print(f"Minimum                   : {p.min()}")
    print(f"Maksimum                  : {p.max()}")
    print(f"Mean                      : {p.mean():.2f}")
    print(f"Median                    : {p.median():.1f}")
    print(f"Standar deviasi           : {p.std():.2f}")

    # --- 8. Tabel 10 panjang tersering ---------------------------------
    top10 = (frek.head(10).rename_axis("jumlah_karakter")
             .reset_index(name="jumlah_mof"))
    top10["persen"] = (100 * top10["jumlah_mof"] / len(p)).round(2)
    print("\n10 PANJANG KARAKTER PALING SERING MUNCUL")
    print(top10.to_string(index=False))

    # --- 5. Histogram --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    lebar_bin = 2
    bins = range(int(p.min()), int(p.max()) + lebar_bin + 1, lebar_bin)
    ax.hist(p, bins=bins, color="#2b4a75", edgecolor="white", linewidth=0.5)

    ax.axvline(p.mean(), color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"mean = {p.mean():.1f}")
    ax.axvline(p.median(), color="#e08c1a", linestyle="-.", linewidth=1.4,
               label=f"median = {p.median():.0f}")
    ax.axvline(dominan, color="#1e8449", linestyle=":", linewidth=1.8,
               label=f"modus = {dominan}")

    ax.set_title("Distribusi Jumlah Karakter String MOFHAPPY",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Jumlah karakter pada kolom `mofhappy`", fontsize=10.5)
    ax.set_ylabel("Jumlah MOF", fontsize=10.5)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)

    fig.tight_layout()
    fig.savefig("histogram_karakter.png")
    print("\nHistogram disimpan: histogram_karakter.png")

    # --- 10. Simpan hasil ----------------------------------------------
    df.to_csv("hasil_mofhappy_analisis.csv", index=False)
    (frek.sort_index().rename_axis("jumlah_karakter")
        .reset_index(name="jumlah_mof")
        .to_csv("frekuensi_panjang.csv", index=False))
    print("Data disimpan    : hasil_mofhappy_analisis.csv")
    print("Frekuensi disimpan: frekuensi_panjang.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hasil_mofhappy.csv")
