# Spesifikasi aturan MOFHAPPY v0.1

## Bentuk string

```
H  <NODE>  ~  <LINKER>  T
```

| Elemen | Fungsi |
|---|---|
| `H` | sentinel awal |
| `<NODE>` | tepat satu token node logam |
| `~` | pemisah segmen |
| `<LINKER>` | ekspresi pohon fragmen |
| `T` | sentinel akhir |

## R1 — Token node dari komposisi

Node dibentuk dari komposisi, bukan dari pencocokan pola SMILES, karena
valensi ikatan pada node MOF tidak lazim dan sering membuat parser gagal.

```
<simbol logam><nuklearitas><motif jembatan>
```

Prosedur: parse dengan `sanitize=False`, hitung frekuensi unsur, pilih logam
terbanyak (tie-break alfabetis), tentukan motif (`O` bila ada oksigen, `N`
bila nitrogen lebih banyak), lalu terapkan alias untuk motif bernama.
Kegagalan parse menghasilkan `M?`, bukan pembatalan.

## R2 — Gugus koordinasi diprioritaskan

Dicari sebelum cincin, karena gugus inilah yang menempel ke logam dan tidak
boleh terserap ke fragmen lain. Kecocokan diproses dalam urutan kanonik
berdasarkan `Chem.CanonicalRankAtoms`, bukan urutan atom internal RDKit.

| Token | Gugus |
|:---:|---|
| `K` | karboksilat |
| `S` | sulfonat |
| `F` | fosfonat |
| `Z` | imidazolat, tetrazolat |
| `W` | N-donor piridil |

Tanpa pengurutan kanonik, dua pola yang bertumpang tindih dapat diselesaikan
berbeda tergantung cara SMILES ditulis — sumber non-determinisme yang nyata
dan terukur.

## R3 — Sistem cincin terfusi sebagai satu unit

Cincin yang berbagi atom digabung menjadi satu fragmen. Naftalena adalah satu
token, bukan dua benzena.

## R4 — Sisa atom menjadi fragmen rantai

Dikelompokkan per komponen terhubung. Seluruhnya karbon → `C<n>`; ada
heteroatom → gabungan simbol unsur terurut.

## R5 — Gugus kecil diserap menjadi modifier

Fragmen beratom tunggal dengan satu tetangga berupa cincin dihapus dan dicatat
sebagai modifier dalam kurung kurawal, diurutkan alfabetis.

| Atom | Modifier |
|:---:|:---:|
| N | `A` |
| C | `M` |
| O | `O` |
| F | `F` |
| Cl | `L` |
| Br | `R` |
| nitro | `X` |

## R6 — Vocabulary cincin data-driven

Token cincin ditentukan dari canonical SMILES sistem cincin, dicari di
`RING_VOCAB`. Tabel dibangun dari statistik frekuensi pada basis data acuan,
dipotong pada cakupan kumulatif 95 %, dan dimuat dari `ring_vocab.csv`
sehingga perluasan tidak memerlukan perubahan kode.

| Target cakupan | Jumlah token |
|---:|---:|
| 80 % | 8 |
| 90 % | 23 |
| **95 %** | **71** |
| 99 % | 448 |

## R7 — Fallback, bukan kegagalan

Sistem cincin tak dikenal tetap membentuk fragmen dengan token `X`. Posisi,
tetangga, dan modifier tetap terekam. Struktur yang sebagian tidak dikenali
tidak dibuang dari dataset.

## R8 — Graf fragmen dibangun sebelum string ditulis

Setiap ikatan diperiksa; ikatan yang kedua ujungnya di fragmen berbeda menjadi
sisi graf. String tidak pernah ditulis langsung dari urutan atom.

## R9 — Akar penelusuran adalah gugus koordinasi

Bila ada beberapa kandidat, seluruhnya dicoba dan hasil terkecil secara
leksikografis yang dipilih.

## R10 — DFS dengan pengurutan rekursif

Cabang diurutkan berdasarkan string hasil serialisasinya sendiri, bukan indeks
atom. Format: tanpa anak → `tok`; satu anak → `tok-anak`; banyak anak → semua
kecuali terakhir dalam kurung, yang terakhir menjadi ekor.

## R11 — Siklus ditandai

Graf fragmen dapat mengandung siklus. Penelusuran memakai himpunan *visited*;
setiap sisi balik dicatat sebagai `%` pada token. Yang terekam adalah jumlah
penutupan, bukan pasangannya — kehilangan informasi yang disengaja dan terukur
pada 1,5 % linker.

## Grammar formal

```
<mofhappy>    ::= "H" WS <node> WS "~" WS <linker> WS "T"
<node>        ::= <logam> <nuklearitas> [ <motif> ] | "M?"
<linker>      ::= <simpul>
<simpul>      ::= <token> { "(" <simpul> ")" } [ "-" <simpul> ]
<token>       ::= <dasar> { "%" } [ <modifier> ]
<dasar>       ::= <koordinasi> | <cincin> | <rantai>
<koordinasi>  ::= "K" | "S" | "F" | "Z" | "W"
<cincin>      ::= "B" [ "o" | "m" ] | "Bc" | "N6" | ... | "X"
<rantai>      ::= "C" [ digit+ ] | simbol_unsur+
<modifier>    ::= "{" huruf+ "}"
```

Grammar ini menjamin sintaks, bukan validitas kimia. Aturan kompatibilitas —
batas jumlah gugus koordinasi per cincin, larangan koordinasi bersambung
langsung — belum diformalkan.
