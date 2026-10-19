# Metodologi

## Penyiapan data

MOFHAPPY tidak membaca CIF. Dekonstruksi node–linker diasumsikan sudah
dilakukan oleh MOFid atau moffragmentor, menghasilkan tabel dengan kolom
`mof`, `nodes`, dan `linkers` (SMILES, dipisah titik bila lebih dari satu).

Pilihan alat dekonstruksi membatasi cakupan: 94,5 % entitas CoRE MOF berhasil
diproses MOFid, 71,8 % oleh moffragmentor.

## Penyaringan artefak

Fragmen tanpa gugus koordinasi dan fragmen di bawah 4 atom berat dibuang
sebagai artefak dekonstruksi. Pada 19.795 MOF, 5.897 fragmen tersaring.

Artefak yang khas: atom bervalensi tak lazim seperti `[c]`, gugus mustahil
seperti `C(=O)OCl`, dan fragmen sangat kecil.

## Pembagian data

Basis data MOF kombinatorial memakai ulang linker secara masif. Pada 30.000
MOF: 33,1 % linker muncul di lebih dari satu MOF, linker terpopuler muncul di
2.816 MOF, dan 94,5 % MOF berbagi setidaknya satu linker dengan MOF lain.

Karena itu split acak maupun per-MOF membocorkan identitas linker ke test set.
Strategi `linkerset` mengelompokkan berdasarkan kombinasi linker lengkap dan
menjamin tidak ada grup yang muncul di dua sisi. Jaminan ini ditegakkan dengan
`assert`, bukan sekadar diasumsikan.

## Penskalaan fitur

Fitur tabular numerik diskalakan memakai statistik dari train saja. Ini
penting ketika fitur tekstural disertakan, karena luas permukaan berentang
ribuan sementara void fraction berentang 0–1.

## Target

Satu MOF memiliki label pada beberapa tekanan. Setiap pasangan (MOF, tekanan)
menjadi satu baris, dengan tekanan masuk sebagai fitur operasional dalam
log₁₀. Baris-baris dari satu MOF selalu berada di sisi split yang sama.

Distribusi target sangat miring kanan (skewness 2,07), sehingga model
cenderung meratakan prediksi. Opsi `--loss huber` dan
`--transformasi-target log1p` tersedia untuk menanganinya.

## Metrik

Dilaporkan: MSE, RMSE, MAE, R², slope regresi, intercept, Pearson r,
Spearman ρ, dan MAPE bila sesuai.

Slope regresi layak diperhatikan sendiri. Slope di bawah 1 berarti model
meratakan prediksi — meremehkan MOF berkapasitas tinggi, yang justru menjadi
target skrining.

## Validasi silang

`--cv K` menjalankan K-fold berbasis grup, melaporkan rata-rata dan simpangan
baku antar-lipatan.

## Hyperparameter

`--tuning N` menjalankan Optuna dengan objektif MSE pada **validation set**.
Test set hanya disentuh sekali di akhir, memakai konfigurasi terbaik.
