[README.md](https://github.com/user-attachments/files/33654376/README.md)
# MOFHAPPY

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023%2B-2e7d32.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Representasi string coarse-grained bergaya HAPPY untuk Metal-Organic Frameworks,
dengan agregasi multi-linker yang permutation-invariant.**

MOFHAPPY menuliskan sebuah MOF sebagai deretan token pendek di level building
block — satu token untuk metal node, satu token untuk tiap sistem cincin, satu
token untuk tiap gugus koordinasi — menggantikan SMILES atom-per-atom.

```
MOF-5       →  H Zn4O ~ K-B-K T
UiO-66      →  H Zr6O ~ K-B-K T
UiO-67      →  H Zr6O ~ K-B-B-K T
UiO-66-NH2  →  H Zr6O ~ K-B{A}-K T
HKUST-1     →  H Cu2P ~ K-B(K)-K T
ZIF-8       →  H Zn1  ~ Z-C T
```

Repositori ini mencakup pipeline whole-MOF yang meng-encode **seluruh** linker
dalam satu struktur, menggabungkannya dengan agregator min-max-average, dan
membuktikan bahwa hasilnya tidak bergantung pada urutan linker.

---

## Isi repositori

```
MOFHAPPY/
├── mofhappy/                    ← paket inti
│   ├── __init__.py                  ekspor encode_mof, encode_node, dll.
│   ├── mofhappy.py                  encoder: SMILES fragmen → string MOFHAPPY
│   └── wholemof_pipeline.py         bangun dataset, latih model, ablasi
│
├── scripts/                     ← alat bantu
│   ├── uji_permutasi.py             uji permutation-invariance, model acak
│   ├── uji_permutasi_terlatih.py    uji permutation-invariance, model terlatih
│   ├── banding_model.py             grafik perbandingan antar-model
│   ├── plot_parity.py               parity plot + statistik regresi
│   └── analisis_karakter.py         distribusi panjang string
│
├── docs/
│   ├── aturan.md                    11 aturan konstruksi + grammar BNF
│   └── metodologi.md                penyiapan data, strategi split, metrik
│
├── res/                         ← tempat gambar untuk README
├── README.md
├── LICENSE                          MIT
├── CITATION.cff                     metadata sitasi
├── requirements.txt
└── .gitignore                       mengecualikan hasil/, *.npz, *.pt, data besar
```

### Apa isi tiap berkas

| Berkas | Fungsi |
|---|---|
| `mofhappy/mofhappy.py` | Mengubah SMILES node dan linker menjadi string MOFHAPPY. Memuat vocabulary cincin, pola SMARTS gugus koordinasi, fragmentasi, dan serialisasi kanonik. |
| `mofhappy/wholemof_pipeline.py` | Empat tahap: `data` membangun dataset, `diagnosa` memeriksanya, `latih` melatih model, `ablasi` menjalankan seluruh kombinasi. Memuat tujuh mode agregasi dan empat strategi split. |
| `scripts/uji_permutasi.py` | Mengacak urutan slot linker pada model berbobot acak, mengukur selisih keluaran. Cepat, untuk memeriksa arsitektur. |
| `scripts/uji_permutasi_terlatih.py` | Sama, tetapi pada checkpoint hasil training dan seluruh test set. Melaporkan ΔR² dan ΔRMSE. Ini bukti utama. |
| `scripts/banding_model.py` | Membaca semua `prediksi_*.csv`, menghasilkan bar chart, grid parity, histogram residual, dan tabel ringkas. |
| `scripts/plot_parity.py` | Parity plot satu model: garis parity, garis regresi, slope, R², RMSE, MAE. |
| `scripts/analisis_karakter.py` | Statistik panjang string dan histogramnya. |

---

## Quickstart

```bash
git clone https://github.com/<user>/MOFHAPPY.git
cd MOFHAPPY
pip install -r requirements.txt
```

### 1. Encode satu struktur

```python
from mofhappy.mofhappy import encode_mof

print(encode_mof("[Zr][O]([Zr])([Zr])([Zr])([Zr])[Zr]",
                 "O=C([O-])c1ccc(C(=O)[O-])cc1"))
# H Zr6O ~ K-B-K T
```

### 2. Bangun dataset

```bash
python mofhappy/wholemof_pipeline.py --tahap data \
       --berkas-linker data_siap.csv --n-mof 20000 --maks-linker 3
```

`data_siap.csv` berisi satu baris per MOF dengan kolom `mof`, `nodes`, dan
`linkers` dalam SMILES, seperti keluaran
[MOFid](https://github.com/snurr-group/mofid) atau
[moffragmentor](https://github.com/kjappelbaum/moffragmentor).

### 3. Periksa dataset

```bash
python mofhappy/wholemof_pipeline.py --tahap diagnosa
```

Melaporkan distribusi jumlah linker, proporsi padding, kemiringan target, dan
perkiraan kebocoran untuk tiap strategi split.

### 4. Latih

```bash
python mofhappy/wholemof_pipeline.py --tahap latih \
       --encoder lstm --agregasi minmaxavg --split linkerset
```

### 5. Buktikan permutation-invariance

```bash
python scripts/uji_permutasi_terlatih.py hasil
```

---

## Representasi

Satu MOF ditulis sebagai dua segmen yang dipisah `~`, diapit sentinel:

```
H  <NODE>  ~  <LINKER>  T
```

| Kelas token | Contoh | Asal |
|---|---|---|
| Metal node | `Zn4O`, `Cu2P`, `Zr6O`, `Cr3O` | komposisi fragmen node |
| Gugus koordinasi | `K` karboksilat, `Z` azolat, `W` N-donor piridil | pencocokan SMARTS |
| Sistem cincin | `B` benzena, `Bc` naftalena, `N6` piridin, `X` tak dikenal | vocabulary dari frekuensi data |
| Rantai | `C2`, `CN`, `CClO` | sisa komponen terhubung |
| Modifier | `{A}` amino, `{FF}` difluoro | gugus terminal beratom tunggal |
| Struktur | `-` ikatan, `( )` cabang, `%` penutup siklus | sintaks penelusuran |

Sebelas aturan bernomor mengatur konstruksinya; spesifikasi lengkap beserta
grammar BNF ada di [`docs/aturan.md`](docs/aturan.md).

### Keringkasan

| | MOFHAPPY | SMILES |
|---|---:|---:|
| Median token per linker | **9** | 38 |
| Ukuran vocabulary | 730 | 41 |
| Panjang maksimum | **44** | 296 |
| Linker terpotong pada panjang praktis | **0 %** | 3,9 % |
| Detik per epoch (CPU, 3.000 MOF) | **68** | 235 |

### Jaminan desain

**Deterministik.** Satu struktur menghasilkan tepat satu string. Diuji pada
1.462 linker × 10 penulisan SMILES acak: **100,00 %** identik.

**Tidak ambigu.** String dapat diurai kembali menjadi graf fragmen yang sama.
Diuji pada 1.945 linker: **100,00 %** round-trip.

**Graceful.** Sistem cincin tak dikenal turun ke token `X`, bukan membatalkan
seluruh struktur.

---

## Whole-MOF multi-linker

Sebagian besar MOF memiliki lebih dari satu linker berbeda. Pada subset hMOF
yang dipakai di sini, hanya 6 % yang benar-benar satu linker; rata-ratanya 2,6.
Meng-encode linker pertama saja membuang sebagian besar informasi kimianya.

Pipeline meng-encode **semua** linker, melewatkan tiap linker melalui encoder
yang **berbagi bobot**, lalu menggabungkannya dengan agregator simetris:

```
MOF → [linker₁, linker₂, linker₃]
         ↓ encoder berbagi bobot (ANN / RNN / LSTM / GRU)
      [h₁, h₂, h₃]
         ↓ agregasi
      z_MOF  →  prediksi
```

| `--agregasi` | Operasi | Permutation-invariant |
|---|---|:---:|
| `single` | hanya linker pertama (baseline lama) | tidak |
| `concat_unshared` | bobot terpisah per slot, concatenate | tidak |
| `concat_shared` | bobot bersama, concatenate | tidak |
| `mean` / `sum` | rata-rata atau jumlah ter-mask | **ya** |
| **`minmaxavg`** | **gabungan min, rata-rata, max ter-mask** | **ya** |
| `deepsets` | φ → Σ → ρ | **ya** |
| `attention` | bobot per linker yang dipelajari | **ya** |

---

## Pembuktian permutation-invariance

Urutan linker dalam sebuah MOF tidak punya makna fisik — ia hanya hasil
pengurutan berkas. Model yang benar seharusnya menghasilkan prediksi identik
ketika urutan itu diacak.

Pengujian dilakukan pada model **terlatih**, dengan mengacak urutan slot linker
lima kali pada seluruh test set, lalu mengukur selisih keluaran dan perubahan
metrik akhir.

```bash
python scripts/uji_permutasi_terlatih.py hasil
```

| Agregasi | Selisih maksimum | ΔR² | ΔRMSE | Status |
|---|---:|---:|---:|---|
| `single` (ANN) | 7,2 × 10⁰ | 8,9 × 10⁻² | 1,8 × 10⁻¹ | VARIANT |
| `single` (RNN) | 6,7 × 10⁰ | 7,8 × 10⁻² | 1,6 × 10⁻¹ | VARIANT |
| `single` (LSTM) | 7,1 × 10⁰ | 5,8 × 10⁻² | 1,2 × 10⁻¹ | VARIANT |
| `concat_shared` | 1,3 × 10⁰ | 6,6 × 10⁻² | 2,6 × 10⁻³ | VARIANT |
| **`minmaxavg` (ANN)** | **9,5 × 10⁻⁷** | **0,00** | **0,00** | **INVARIANT** |
| **`minmaxavg` (RNN)** | **1,9 × 10⁻⁶** | **0,00** | 4,6 × 10⁻⁸ | **INVARIANT** |
| **`minmaxavg` (LSTM)** | **9,5 × 10⁻⁷** | **0,00** | 4,7 × 10⁻⁸ | **INVARIANT** |

Selisih di orde 10⁻⁶ adalah presisi floating-point, bukan ketergantungan
urutan. Bukti yang paling tegas adalah **ΔR² dan ΔRMSE tepat nol** — metrik
akhirnya identik sampai digit yang dilaporkan.

### Temuan: berbagi bobot saja tidak cukup

`concat_shared` sudah memakai encoder yang berbagi bobot, tetapi tetap VARIANT.
Penyebabnya bukan encoder, melainkan operasi penggabungannya: menyambung
`[h₁, h₂, h₃]` secara berurutan membuat posisi slot punya makna di lapisan
berikutnya.

Yang menghilangkan ketergantungan urutan adalah **agregator simetrisnya** —
min, rata-rata, dan max tidak peduli urutan masukan. Keduanya hal yang terpisah:

| | Dampak |
|---|---|
| Berbagi bobot | efisiensi parameter (turun 64 %) |
| Agregasi simetris | permutation-invariance |

### Tanpa mengorbankan akurasi

| Encoder | `single` | `concat_shared` | `minmaxavg` |
|---|---:|---:|---:|
| ANN | 0,191 | 0,302 | 0,290 |
| RNN | 0,202 | 0,306 | **0,320** |
| LSTM | 0,214 | 0,312 | **0,318** |

`minmaxavg` setara `concat_shared` (selisih ≤ 0,012), sehingga
permutation-invariance diperoleh tanpa biaya akurasi. Agregasi multi-linker
sendiri menyumbang **+0,10 R²** dibanding encoding satu linker, konsisten di
ketiga arsitektur.

---

## Pembagian data sadar-kebocoran

Basis data MOF kombinatorial memakai ulang linker secara masif: 94,5 % struktur
berbagi setidaknya satu linker dengan struktur lain, dan linker terpopuler
muncul di 2.816 di antaranya. Split acak maupun per-MOF karenanya membocorkan
identitas linker ke test set.

| `--split` | Kunci pengelompokan | Keketatan |
|---|---|---|
| `random` | baris | bocor |
| `mof` | identitas MOF | sebagian |
| `linkerset` | kombinasi linker lengkap | **disarankan** |
| `linker` | linker dominan | paling ketat |

Splitter menegakkan jaminan dengan `assert` bahwa tidak ada grup muncul di dua
sisi, dan melaporkan berapa linker test yang sudah terlihat saat training.

---

## Requirements

```
python >= 3.9
rdkit >= 2023.3
torch >= 2.0
pandas, numpy, scipy, matplotlib
```

---

## Batasan

- Topologi sengaja berada di luar lingkup. String meng-encode identitas kimia,
  bukan geometri pori; segmen `<NET>` dapat ditambahkan tanpa mengubah dua
  segmen pertama.
- Penutup siklus mencatat jumlahnya, bukan pasangan fragmen yang ditutup.
  Memengaruhi 1,5 % linker.
- Posisi substituen tidak terekam, sehingga isomer posisi tidak terbedakan.
- Cakupan bergantung pada alat dekonstruksi: 94,5 % CoRE MOF terurai dengan
  MOFid, 71,8 % dengan moffragmentor.
- Divalidasi pada hMOF. CoRE MOF memiliki keragaman node dan linker yang jauh
  lebih besar dan belum diuji.

---

## Sitasi

```bibtex
@software{mofhappy,
  title  = {MOFHAPPY: representasi string coarse-grained untuk
            metal-organic frameworks},
  author = {<Achmad Nabawi Dewa Wardhana>},
  year   = {2026},
  url    = {https://github.com/<user>/MOFHAPPY}
}
```

## Acknowledgements

Desainnya mengikuti representasi HAPPY untuk polimer
([Ahn dkk., npj Comput. Mater. 2024](https://doi.org/10.1038/s41524-024-01293-8)),
dibangun di atas MOFid
([Bucior dkk., Cryst. Growth Des. 2019](https://doi.org/10.1021/acs.cgd.9b01050)),
dan meminjam skema simbol building block dari
[pyCOFBuilder](https://arxiv.org/abs/2310.14822).

## Lisensi

MIT — lihat [LICENSE](LICENSE).
