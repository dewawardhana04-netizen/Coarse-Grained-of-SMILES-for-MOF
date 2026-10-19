"""
wholemof_pipeline.py
==============================================================================
Whole-MOF multi-linker representation untuk MOFHAPPY.

PERUBAHAN INTI DARI PIPELINE LAMA
  Lama : satu MOF -> satu linker (linkers.split(".")[0])  -> sekuens tunggal
  Baru : satu MOF -> HIMPUNAN semua linker                -> set of sequences
         digabung dengan agregasi permutation-invariant (mean / attention /
         Deep Sets), sehingga urutan linker di berkas TIDAK memengaruhi hasil.

MODE AGREGASI (untuk ablation study)
  single    : hanya linker pertama            <- baseline lama
  mean      : rata-rata ter-mask semua linker  <- multi-linker sederhana
  sum       : jumlah ter-mask (peka multiplicity)
  attention : attention pooling berbobot       <- bobot per linker dipelajari
  deepsets  : phi -> sum -> rho (Deep Sets)

STRATEGI SPLIT (untuk menguji leakage)
  random      : acak per baris                     <- BOCOR, hanya pembanding
  mof         : per identitas MOF                  <- yang dipakai sebelumnya
  linkerset   : per kombinasi linker (group-aware) <- paling ketat
  linker      : per linker dominan (scaffold-like)

Pakai:
    python3 wholemof_pipeline.py --tahap data   --n-mof 20000
    python3 wholemof_pipeline.py --tahap diagnosa
    python3 wholemof_pipeline.py --tahap latih  --agregasi attention --split linkerset
    python3 wholemof_pipeline.py --tahap ablasi
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("wholemof")

PAD_ID, UNK_ID = 0, 1
PAD, UNK = "[PAD]", "[UNK]"

TEKANAN = {"CO2_capacity_001": 0.01, "CO2_capacity_005": 0.05,
           "CO2_capacity_01": 0.1, "CO2_capacity_05": 0.5,
           "CO2_capacity_25": 2.5}


# ============================================================ KONFIGURASI
@dataclass
class Konfig:
    # -- berkas (tidak ada path hard-coded di badan kode) --
    dir_data: Path = Path(".")
    berkas_linker: str = "data_siap.csv"
    berkas_label: str = "hMOF_CO2_info.csv"
    # Fitur tabular tekstural (opsional): void fraction, luas permukaan
    # gravimetrik & volumetrik, PLD, LCD. Kosongkan bila tidak ada.
    berkas_tabular: str = ""
    kolom_tabular: tuple = ()          # kosong = pakai semua kolom numerik
    tabular_murni: bool = False        # True = HANYA fitur tekstural
                                       # (buang node one-hot & jumlah linker)
    cv: int = 0                        # >0 = K-fold cross validation
    n_target: int = 0                  # >0 = batasi jumlah MOF (mis. 8423)
    dir_hasil: Path = Path("hasil")
    cache: str = "wholemof_data.npz"   # ditimpa otomatis per representasi

    # -- pembentukan dataset --
    n_mof: Optional[int] = 20000        # None = semua
    maks_linker: int = 3                # slot linker per MOF (3 paling dominan)
    maks_token: int = 42                # panjang token per linker (MOFHAPPY)
    representasi: str = "mofhappy"      # mofhappy|smiles  <- untuk baseline
    maks_token_smiles: int = 96         # SMILES jauh lebih panjang
    batas_char: int = 400               # batas string MOFHAPPY
    buang_artefak: bool = True          # buang fragmen tanpa gugus koordinasi
    pertahankan_multiplicity: bool = True

    # -- model --
    encoder: str = "gru"                # ann|rnn|lstm|gru
    tanpa_sekuens: bool = False         # True = baseline fitur tabular SAJA
    tekanan: float = None               # None = semua tekanan digabung
    agregasi: str = "minmaxavg"
    # single|mean|sum|minmaxavg|max|min|attention|deepsets
    # concat_unshared : encoder TERPISAH per slot + concatenate  (bobot beda)
    # concat_shared   : encoder BERSAMA + concatenate            (bobot sama)
    embedding_dim: int = 64
    hidden: int = 128
    head_units: int = 256
    dropout: float = 0.15
    loss: str = "mse"                   # mse|huber
    transformasi_target: str = "none"   # none|log1p

    # -- training --
    lr: float = 1e-3
    batch: int = 256
    epochs: int = 250
    patience: int = 30
    seed: int = 42

    # -- split --
    split: str = "linkerset"            # random|mof|linkerset|linker
    val_frac: float = 0.15
    test_frac: float = 0.15


# ============================================================ TOKENISASI
def tokenize(s: str) -> List[str]:
    """Pecah string MOFHAPPY menjadi token; modifier {..} menempel."""
    out, buf, i = [], "", 0
    while i < len(s):
        c = s[i]
        if c == "{":
            j = s.find("}", i)
            j = len(s) - 1 if j == -1 else j
            buf += s[i:j + 1]
            i = j + 1
            continue
        if c in "-()~ ":
            if buf:
                out.append(buf); buf = ""
            if c != " ":
                out.append(c)
        else:
            buf += c
        i += 1
    if buf:
        out.append(buf)
    return out


_RE_SMILES = re.compile(
    r"(\[[^\]]+]|Br|Cl|Si|Se|se|@@|@|=|#|\-|\+|\\|/|:|~|%[0-9]{2}|[0-9]|"
    r"[A-Za-z]|\(|\)|\.)")


def tokenize_smiles(s: str) -> List[str]:
    """Tokenizer SMILES standar (skema Schwaller/atom-level).

    Dipakai sebagai BASELINE tanpa coarse-graining: setiap atom, ikatan,
    kurung, dan penomoran cincin menjadi token tersendiri.
    """
    return _RE_SMILES.findall(s)


# ============================================================ TAHAP 1: DATA
def bangun_dataset(cfg: Konfig) -> None:
    """Encode SELURUH linker tiap MOF menjadi himpunan sekuens token."""
    from mofhappy import encode_node, encode_linker, COORD_SMARTS  # noqa
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    cfg.dir_hasil.mkdir(parents=True, exist_ok=True)
    src = cfg.dir_data / cfg.berkas_linker
    if not src.exists():
        raise FileNotFoundError(f"{src} tidak ditemukan.")

    df = pd.read_csv(src)
    log.info("Dibaca %s: %d MOF", src.name, len(df))
    if cfg.n_mof:
        df = df.sample(min(cfg.n_mof, len(df)), random_state=cfg.seed)
        log.info("Disampel menjadi %d MOF (seed=%d)", len(df), cfg.seed)

    pola_koord = [Chem.MolFromSmarts(s) for _, s in COORD_SMARTS]
    pola_koord = [p for p in pola_koord if p is not None]

    def linker_valid(smi: str) -> bool:
        """Artefak dekonstruksi: fragmen tanpa gugus koordinasi apa pun."""
        if not cfg.buang_artefak:
            return True
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumHeavyAtoms() < 4:
            return False
        return any(m.HasSubstructMatch(p) for p in pola_koord)

    baris, n_buang_art, n_gagal = [], 0, 0
    t0 = time.time()
    for k, (_, r) in enumerate(df.iterrows(), 1):
        if k % 5000 == 0:
            log.info("  encode %d/%d (%.0fs)", k, len(df), time.time() - t0)
        frag = str(r["linkers"]).split(".")
        # multiplicity: pertahankan salinan, atau ambil unik saja
        if not cfg.pertahankan_multiplicity:
            frag = sorted(set(frag))
        dipakai = []
        for smi in frag:
            if not linker_valid(smi):
                n_buang_art += 1
                continue
            if cfg.representasi == "smiles":
                # BASELINE: SMILES apa adanya, tanpa coarse-graining
                s = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            else:
                s = encode_linker(smi)
                if s is None:
                    n_gagal += 1
                    continue
                if len(s) > cfg.batas_char:
                    continue
            dipakai.append(s)
        if not dipakai:
            continue
        # urutkan kanonik supaya berkas reproducible (model tetap
        # permutation-invariant, tetapi berkasnya deterministik)
        dipakai = sorted(dipakai)
        baris.append({
            "mof": r["mof"],
            "node": encode_node(str(r["nodes"]).split(".")[0]),
            "linkers_happy": "|".join(dipakai),
            "n_linker_asli": len(frag),
            "n_linker_dipakai": len(dipakai),
            "n_linker_unik": len(set(dipakai)),
        })

    enc = pd.DataFrame(baris)
    log.info("MOF ter-encode: %d | fragmen artefak dibuang: %d | gagal: %d",
             len(enc), n_buang_art, n_gagal)
    if enc.empty:
        raise SystemExit("Tidak ada MOF yang berhasil di-encode.")

    # ---- vocabulary token (hanya dari data ini) ----
    tok_fn = tokenize_smiles if cfg.representasi == "smiles" else tokenize
    maks_tok = (cfg.maks_token_smiles if cfg.representasi == "smiles"
                else cfg.maks_token)
    hit = Counter()
    panjang_tok = []
    for s in enc["linkers_happy"]:
        for l in s.split("|"):
            t = tok_fn(l)
            hit.update(t); panjang_tok.append(len(t))
    tok2id = {PAD: PAD_ID, UNK: UNK_ID}
    for t, _ in hit.most_common():
        tok2id.setdefault(t, len(tok2id))
    node_vocab = sorted(enc["node"].unique())
    n2i = {n: i for i, n in enumerate(node_vocab)}
    pt = np.array(panjang_tok)
    log.info("Representasi=%s | vocabulary token: %d | jenis node: %d",
             cfg.representasi, len(tok2id), len(node_vocab))
    log.info("Panjang token per linker: median %d | mean %.1f | maks %d "
             "| p99 %d  (maks_token dipakai = %d)",
             int(np.median(pt)), pt.mean(), pt.max(),
             int(np.percentile(pt, 99)), maks_tok)
    if (pt > maks_tok).mean() > 0.01:
        log.warning("%.1f%% linker terpotong pada maks_token=%d",
                    100 * (pt > maks_tok).mean(), maks_tok)

    # ---- fitur tabular tekstural (opsional) ----
    kol_tekstural = []
    if cfg.berkas_tabular:
        ft = cfg.dir_data / cfg.berkas_tabular
        if not ft.exists():
            raise FileNotFoundError(f"{ft} tidak ditemukan.")
        tab_df = pd.read_csv(ft) if ft.suffix.lower() == ".csv" \
            else pd.read_excel(ft)
        kol_id_t = next((c for c in ("mof", "MOF", "name", "id")
                         if c in tab_df.columns), None)
        if kol_id_t is None:
            raise SystemExit(f"Kolom identitas MOF tidak ada di {ft.name}. "
                             f"Kolom: {list(tab_df.columns)}")
        if cfg.kolom_tabular:
            kol_tekstural = [c for c in cfg.kolom_tabular if c in tab_df.columns]
            hilang = [c for c in cfg.kolom_tabular if c not in tab_df.columns]
            if hilang:
                raise SystemExit(f"Kolom tabular tidak ada: {hilang}")
        else:
            kol_tekstural = [c for c in tab_df.columns
                             if c != kol_id_t
                             and pd.api.types.is_numeric_dtype(tab_df[c])]
        n_sebelum = len(enc)
        enc = enc.merge(tab_df[[kol_id_t] + kol_tekstural],
                        left_on="mof", right_on=kol_id_t, how="inner")
        if kol_id_t != "mof":
            enc = enc.drop(columns=[kol_id_t])
        log.info("Fitur tabular dari %s: %s", ft.name, kol_tekstural)
        log.info("MOF cocok fitur tabular: %d / %d (%.1f%%)",
                 len(enc), n_sebelum, 100 * len(enc) / max(n_sebelum, 1))
        n_na = enc[kol_tekstural].isna().sum().sum()
        if n_na:
            log.warning("%d nilai kosong pada fitur tabular -> diisi median "
                        "(median dihitung dari data ini)", n_na)
            enc[kol_tekstural] = enc[kol_tekstural].fillna(
                enc[kol_tekstural].median())
        if enc.empty:
            raise SystemExit("Tidak ada MOF yang cocok dengan berkas tabular.")

    # ---- batasi jumlah MOF agar setara penelitian pembanding ----
    if cfg.n_target and len(enc) > cfg.n_target:
        enc = enc.sample(cfg.n_target, random_state=cfg.seed
                         ).reset_index(drop=True)
        log.info("Dibatasi menjadi %d MOF (n_target, seed=%d)",
                 len(enc), cfg.seed)

    # ---- label ----
    lab = pd.read_csv(cfg.dir_data / cfg.berkas_label)
    kol_t = [c for c in TEKANAN if c in lab.columns]
    if not kol_t:
        raise SystemExit("Kolom target tidak ditemukan di berkas label.")
    enc = enc.merge(lab[["MOF"] + kol_t], left_on="mof", right_on="MOF",
                    how="inner").drop(columns=["MOF"])
    log.info("MOF dengan label: %d | kolom target: %s", len(enc), kol_t)

    # ---- tensorisasi: [N, maks_linker, maks_token] ----
    def enc_seq(s: str) -> List[int]:
        ids = [tok2id.get(t, UNK_ID) for t in tok_fn(s)][:maks_tok]
        return ids + [PAD_ID] * (maks_tok - len(ids))

    SEQ, LMASK, TAB, PRES, Y, MOF, GRP_LS, GRP_LK = [], [], [], [], [], [], [], []
    n_potong_linker = 0
    for _, r in enc.iterrows():
        ls = r["linkers_happy"].split("|")
        if len(ls) > cfg.maks_linker:
            n_potong_linker += 1
        ls_pakai = ls[:cfg.maks_linker]
        seq = [enc_seq(x) for x in ls_pakai]
        lmask = [1.0] * len(seq)
        while len(seq) < cfg.maks_linker:          # slot kosong = seluruhnya PAD
            seq.append([PAD_ID] * maks_tok)
            lmask.append(0.0)
        if cfg.tabular_murni:
            # HANYA fitur tekstural, meniru baseline tabular Hagel.
            tab = [float(r[c]) for c in kol_tekstural]
        else:
            onehot = [0.0] * len(node_vocab)
            onehot[n2i[r["node"]]] = 1.0
            tab = onehot + [float(r["n_linker_dipakai"]),
                            float(r["n_linker_unik"])]
            tab += [float(r[c]) for c in kol_tekstural]
        kunci_ls = "|".join(sorted(set(ls)))       # grup: kombinasi linker
        kunci_lk = sorted(set(ls))[0]              # grup: linker dominan
        for c in kol_t:
            if pd.isna(r[c]):
                continue
            SEQ.append(seq); LMASK.append(lmask); TAB.append(tab)
            PRES.append([np.log10(TEKANAN[c])])
            Y.append(float(r[c])); MOF.append(r["mof"])
            GRP_LS.append(kunci_ls); GRP_LK.append(kunci_lk)

    SEQ = np.array(SEQ, dtype=np.int64)
    LMASK = np.array(LMASK, dtype=np.float32)
    TAB = np.array(TAB, dtype=np.float32)
    PRES = np.array(PRES, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    log.info("SEQ %s | LMASK %s | TAB %s | Y %s",
             SEQ.shape, LMASK.shape, TAB.shape, Y.shape)
    if n_potong_linker:
        log.warning("%d MOF punya linker lebih dari maks_linker=%d "
                    "-> kelebihannya dibuang", n_potong_linker, cfg.maks_linker)

    np.savez_compressed(
        cfg.dir_hasil / cfg.cache,
        seq=SEQ, lmask=LMASK, tab=TAB, pres=PRES, y=Y,
        mof=np.array(MOF), grp_ls=np.array(GRP_LS), grp_lk=np.array(GRP_LK))
    # Indeks kolom tabular yang BERSIFAT NUMERIK (bukan one-hot node).
    # Penskalaannya dilakukan di tahap latih memakai statistik TRAIN saja,
    # supaya tidak terjadi kebocoran informasi dari val/test.
    idx_numerik = ([] if cfg.tabular_murni else list(range(len(node_vocab),
                                                          TAB.shape[1])))
    if cfg.tabular_murni:
        idx_numerik = list(range(TAB.shape[1]))     # semuanya numerik
        log.info("MODE TABULAR MURNI: hanya %d fitur tekstural "
                 "(node one-hot & jumlah linker dibuang)", TAB.shape[1])
    json.dump({"tok2id": tok2id, "node_vocab": node_vocab,
               "kolom_tekstural": kol_tekstural,
               "tabular_murni": cfg.tabular_murni,
               "idx_tab_numerik": idx_numerik,
               "representasi": cfg.representasi,
               "maks_linker": cfg.maks_linker, "maks_token": maks_tok,
               "median_token_per_linker": int(np.median(pt)),
               "vocab_size": len(tok2id), "tab_dim": TAB.shape[1],
               "kolom_target": kol_t, "konfig": {k: str(v) for k, v in
                                                 asdict(cfg).items()}},
              open(cfg.dir_hasil / f"meta_{cfg.representasi}.json", "w"),
              indent=2)
    enc.to_csv(cfg.dir_hasil / f"encoded_{cfg.representasi}.csv", index=False)
    log.info("Disimpan ke %s", cfg.dir_hasil)


# ============================================================ TAHAP 2: DIAGNOSA
def diagnosa(cfg: Konfig) -> None:
    d = np.load(cfg.dir_hasil / cfg.cache, allow_pickle=True)
    meta = json.load(open(cfg.dir_hasil / f"meta_{cfg.representasi}.json"))
    seq, lmask, y = d["seq"], d["lmask"], d["y"]
    mof, grp_ls = d["mof"], d["grp_ls"]

    print("=" * 74)
    print("DIAGNOSA DATASET")
    print("=" * 74)
    print(f"Baris (MOF x tekanan) : {len(y):,}")
    print(f"MOF unik              : {len(np.unique(mof)):,}")
    print(f"Bentuk SEQ            : {seq.shape}  "
          f"[N, maks_linker={seq.shape[1]}, maks_token={seq.shape[2]}]")
    print(f"Vocabulary            : {meta['vocab_size']:,}")

    n_link = lmask.sum(1)
    print("\n-- Distribusi jumlah linker per MOF --")
    for k, v in sorted(Counter(n_link.astype(int)).items()):
        bar = "#" * int(60 * v / len(n_link))
        print(f"  {k} linker : {v:>7,} ({100*v/len(n_link):>5.1f}%) {bar}")
    print(f"  rata-rata: {n_link.mean():.2f}")

    isi = (seq != PAD_ID).sum()
    slot_aktif = lmask.sum() * seq.shape[2]
    print("\n-- Padding --")
    print(f"  Token terisi        : {isi:,}")
    print(f"  Slot linker aktif   : {int(lmask.sum()):,} / {lmask.size:,} "
          f"({100*lmask.mean():.1f}%)")
    print(f"  Padding token dalam slot aktif: "
          f"{100*(1-isi/max(slot_aktif,1)):.1f}%")

    print("\n-- Distribusi target --")
    print(f"  min {y.min():.3f} | q25 {np.percentile(y,25):.3f} | "
          f"median {np.median(y):.3f} | q75 {np.percentile(y,75):.3f} | "
          f"maks {y.max():.3f}")
    print(f"  mean {y.mean():.3f} | std {y.std():.3f} | "
          f"skewness {float(pd.Series(y).skew()):.3f}")
    if float(pd.Series(y).skew()) > 1.0:
        print("  -> Sangat miring kanan. Pertimbangkan "
              "--transformasi-target log1p atau --loss huber.")

    print("\n-- RISIKO KEBOCORAN (penting) --")
    n_mof = len(np.unique(mof))
    n_ls = len(np.unique(grp_ls))
    print(f"  MOF unik                        : {n_mof:,}")
    print(f"  Kombinasi linker unik           : {n_ls:,}")
    print(f"  Rasio                           : {n_ls/n_mof:.3f}")
    if n_ls < n_mof * 0.98:
        print(f"  -> {n_mof-n_ls:,} MOF memiliki kombinasi linker yang sama")
        print("     dengan MOF lain. Split per-MOF TIDAK cukup;")
        print("     gunakan --split linkerset.")
    else:
        print("  -> Kombinasi linker hampir seluruhnya unik.")


# ============================================================ MODEL
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def bangun_model(cfg: Konfig, vocab_size: int, tab_dim: int,
                 maks_token: int = None):
    maks_token = maks_token or cfg.maks_token
    torch, nn, F = _torch()

    class EncoderLinker(nn.Module):
        """Encoder yang DIBAGI untuk semua linker (weight sharing).

        Bobot dibagi supaya linker ke-1 dan ke-3 diperlakukan sama —
        prasyarat agar keseluruhan model permutation-invariant.

        cfg.encoder memilih arsitektur pembaca sekuens token:
          ann  : embedding di-mask lalu diratakan (flatten) -> MLP
          rnn  : RNN bidirectional + masked mean-pool
          lstm : LSTM bidirectional + masked mean-pool
          gru  : GRU bidirectional + masked mean-pool
        Semuanya memakai embedding padding_idx=0 dan masking token,
        sehingga posisi padding tidak pernah menyumbang informasi.
        """

        def __init__(self):
            super().__init__()
            self.jenis = cfg.encoder
            self.emb = nn.Embedding(vocab_size, cfg.embedding_dim,
                                    padding_idx=PAD_ID)
            if self.jenis == "ann":
                self.mlp = nn.Sequential(
                    nn.Linear(maks_token * cfg.embedding_dim, cfg.hidden * 2),
                    nn.ReLU(), nn.Dropout(cfg.dropout))
                self.out_dim = cfg.hidden * 2
            else:
                kelas = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.jenis]
                kw = dict(input_size=cfg.embedding_dim, hidden_size=cfg.hidden,
                          num_layers=1, batch_first=True, bidirectional=True)
                if self.jenis == "rnn":
                    kw["nonlinearity"] = "tanh"
                self.rnn = kelas(**kw)
                self.out_dim = cfg.hidden * 2

        def forward(self, ids):                       # [B*L, T]
            e = self.emb(ids)
            tmask = (ids != PAD_ID).float().unsqueeze(-1)
            if self.jenis == "ann":
                e = e * tmask                          # padding -> 0
                return self.mlp(e.reshape(e.size(0), -1))
            h, _ = self.rnn(e)
            h = h * tmask                              # padding token -> 0
            return h.sum(1) / tmask.sum(1).clamp(min=1)   # masked mean

    class AgregasiSet(nn.Module):
        """Gabungkan himpunan vektor linker menjadi satu vektor MOF."""

        def __init__(self, dim):
            super().__init__()
            self.mode = cfg.agregasi
            self.dim = dim
            # dimensi keluaran: minmaxavg menggabungkan 3 statistik
            if self.mode == "minmaxavg":
                self.out_dim = dim * 3
            elif self.mode.startswith("concat"):
                self.out_dim = dim * cfg.maks_linker
            else:
                self.out_dim = dim
            if self.mode == "attention":
                self.skor = nn.Sequential(
                    nn.Linear(dim, dim // 2), nn.Tanh(),
                    nn.Linear(dim // 2, 1))
            elif self.mode == "deepsets":
                self.phi = nn.Sequential(
                    nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                self.rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

        def forward(self, h, lmask):                   # h [B, L, D]
            m = lmask.unsqueeze(-1)
            if self.mode == "single":
                return h[:, 0, :]                      # baseline lama
            if self.mode.startswith("concat"):
                # Gabung berurutan -> POSISI punya makna, jadi hasilnya
                # BERGANTUNG URUTAN (permutation-variant). Ini disengaja,
                # sebagai pembanding sebelum agregasi simetris.
                return (h * m).reshape(h.size(0), -1)
            if self.mode == "mean":
                return (h * m).sum(1) / m.sum(1).clamp(min=1)
            if self.mode == "sum":
                return (h * m).sum(1)
            if self.mode in ("minmaxavg", "max", "min"):
                # Pooling statistik terhadap HIMPUNAN linker.
                # Slot kosong diisi +/-inf agar tidak ikut terpilih.
                mx = h.masked_fill(m == 0, -1e9).max(1).values
                mn = h.masked_fill(m == 0, 1e9).min(1).values
                av = (h * m).sum(1) / m.sum(1).clamp(min=1)
                if self.mode == "max":
                    return mx
                if self.mode == "min":
                    return mn
                return torch.cat([mx, av, mn], dim=1)
            if self.mode == "deepsets":
                return self.rho((self.phi(h) * m).sum(1))
            # attention
            s = self.skor(h).masked_fill(m == 0, -1e9)  # slot kosong diabaikan
            a = torch.softmax(s, dim=1)
            return (h * a).sum(1)

        def bobot(self, h, lmask):
            """Bobot attention per linker (untuk interpretability)."""
            if self.mode != "attention":
                return None
            m = lmask.unsqueeze(-1)
            s = self.skor(h).masked_fill(m == 0, -1e9)
            return torch.softmax(s, dim=1).squeeze(-1)

    class WholeMOF(nn.Module):
        def __init__(self):
            super().__init__()
            self.tanpa_sekuens = cfg.tanpa_sekuens
            if self.tanpa_sekuens:
                # Baseline pembanding: string MOFHAPPY TIDAK dipakai sama
                # sekali; model hanya melihat fitur tabular + tekanan.
                dim = tab_dim + 1
                self.head = nn.Sequential(
                    nn.Linear(dim, cfg.head_units), nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.head_units, cfg.head_units // 2), nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.head_units // 2, 1))
                return
            self.unshared = cfg.agregasi == "concat_unshared"
            if self.unshared:
                # bobot BERBEDA untuk tiap slot linker
                self.enc_list = nn.ModuleList(
                    [EncoderLinker() for _ in range(cfg.maks_linker)])
                self.enc = self.enc_list[0]
            else:
                self.enc = EncoderLinker()          # bobot DIBAGI
            self.agg = AgregasiSet(self.enc.out_dim)
            if not hasattr(self.agg, "out_dim"):
                self.agg.out_dim = self.enc.out_dim
            dim = self.agg.out_dim + tab_dim + 1        # +1 tekanan
            self.head = nn.Sequential(
                nn.Linear(dim, cfg.head_units), nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_units, cfg.head_units // 2), nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_units // 2, 1))

        def forward(self, seq, lmask, tab, pres, kembalikan_bobot=False):
            if self.tanpa_sekuens:
                return self.head(torch.cat([tab, pres], dim=1)).squeeze(-1)
            B, L, T = seq.shape
            if self.unshared:
                h = torch.stack([self.enc_list[i](seq[:, i, :])
                                 for i in range(L)], dim=1)
            else:
                h = self.enc(seq.reshape(B * L, T)).reshape(B, L, -1)
            z = self.agg(h, lmask)
            out = self.head(torch.cat([z, tab, pres], dim=1)).squeeze(-1)
            if kembalikan_bobot:
                return out, self.agg.bobot(h, lmask)
            return out

    return WholeMOF()


# ============================================================ SPLIT
def buat_split(cfg: Konfig, d) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kunci = {"random": np.arange(len(d["y"])).astype(str),
             "mof": d["mof"], "linkerset": d["grp_ls"],
             "linker": d["grp_lk"]}[cfg.split]
    rng = np.random.default_rng(cfg.seed)
    unik = np.unique(kunci); rng.shuffle(unik)
    n = len(unik)
    n_te, n_va = int(n * cfg.test_frac), int(n * cfg.val_frac)
    s_te = set(unik[:n_te].tolist())
    s_va = set(unik[n_te:n_te + n_va].tolist())
    is_te = np.array([k in s_te for k in kunci])
    is_va = np.array([k in s_va for k in kunci])
    is_tr = ~(is_te | is_va)

    for a, b, nm in ((is_tr, is_va, "train/val"), (is_tr, is_te, "train/test"),
                     (is_va, is_te, "val/test")):
        assert not (set(kunci[a]) & set(kunci[b])), f"Kebocoran {nm}"

    log.info("Split '%s': grup train %d | val %d | test %d",
             cfg.split, n - n_va - n_te, n_va, n_te)
    log.info("           baris train %d | val %d | test %d",
             is_tr.sum(), is_va.sum(), is_te.sum())
    # diagnostik kebocoran sisa: berapa MOF test yang linkernya pernah dilihat
    if cfg.split in ("mof", "random"):
        set_tr = set()
        for s in d["grp_ls"][is_tr]:
            set_tr.update(str(s).split("|"))
        bocor = sum(1 for s in d["grp_ls"][is_te]
                    if set(str(s).split("|")) & set_tr)
        log.warning("Linker test yang SUDAH pernah muncul di train: %d/%d "
                    "(%.1f%%) -> R2 akan optimistis",
                    bocor, is_te.sum(), 100 * bocor / max(is_te.sum(), 1))
    return is_tr, is_va, is_te


# ============================================================ METRIK
def metrik(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    from scipy import stats
    res = pred - true
    mse = float((res ** 2).mean())
    ss = float(((true - true.mean()) ** 2).sum())
    slope, intercept = np.polyfit(true, pred, 1)
    out = {"MSE": mse, "RMSE": float(np.sqrt(mse)),
           "MAE": float(np.abs(res).mean()),
           "R2": float(1 - (res ** 2).sum() / ss),
           "slope": float(slope), "intercept": float(intercept)}
    try:
        out["pearson"] = float(stats.pearsonr(true, pred)[0])
        out["spearman"] = float(stats.spearmanr(true, pred)[0])
    except Exception:
        pass
    nz = true > 1e-6
    if nz.sum() > 0.5 * len(true):
        out["MAPE"] = float(np.abs(res[nz] / true[nz]).mean() * 100)
    return out


# ============================================================ TAHAP 3: LATIH
def buat_lipatan(cfg: Konfig, d, k: int):
    """K-fold berbasis GRUP, meniru k-fold cross validation Hagel.

    Pengelompokan tetap memakai cfg.split supaya MOF dengan kombinasi
    linker sama tidak terpecah antar-lipatan.
    """
    kunci = {"random": np.arange(len(d["y"])).astype(str),
             "mof": d["mof"], "linkerset": d["grp_ls"],
             "linker": d["grp_lk"]}[cfg.split]
    rng = np.random.default_rng(cfg.seed)
    unik = np.unique(kunci); rng.shuffle(unik)
    potong = np.array_split(unik, k)
    for i in range(k):
        set_te = set(potong[i].tolist())
        sisa = np.concatenate([potong[j] for j in range(k) if j != i])
        n_va = max(1, int(len(sisa) * 0.15))
        set_va = set(sisa[:n_va].tolist())
        is_te = np.array([x in set_te for x in kunci])
        is_va = np.array([x in set_va for x in kunci])
        yield i + 1, ~(is_te | is_va), is_va, is_te


def latih_cv(cfg: Konfig) -> Dict:
    """Jalankan K-fold CV lalu laporkan rata-rata +/- simpangan baku."""
    import copy
    d = dict(np.load(cfg.dir_hasil / cfg.cache, allow_pickle=True))
    # Lipatan harus dibuat pada data yang SUDAH tersaring tekanan,
    # karena latih() juga menyaringnya dengan cara yang sama.
    if cfg.tekanan is not None:
        target = float(np.log10(cfg.tekanan))
        pilih = np.abs(d["pres"][:, 0] - target) < 1e-6
        d = {k: (v[pilih] if hasattr(v, "__len__") and len(v) == len(pilih)
                 else v) for k, v in d.items()}
        log.info("Filter tekanan %.2f bar -> %d baris untuk CV",
                 cfg.tekanan, int(pilih.sum()))
    log.info("=" * 60)
    log.info("CROSS VALIDATION %d-FOLD (grup: %s)", cfg.cv, cfg.split)
    log.info("=" * 60)
    kumpul = []
    for i, tr, va, te in buat_lipatan(cfg, d, cfg.cv):
        c = copy.copy(cfg)
        c.cv = 0
        c._lipatan = (tr, va, te)
        log.info("--- lipatan %d/%d ---", i, cfg.cv)
        r = latih(c, diam=True)
        t = r["hasil"]["test"]
        log.info("    R2 %.4f | RMSE %.4f | MAE %.4f",
                 t["R2"], t["RMSE"], t["MAE"])
        kumpul.append(t)

    ring = {}
    for m in ("R2", "RMSE", "MAE", "slope"):
        v = np.array([x[m] for x in kumpul])
        ring[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    print(f"\n{'metrik':<10}{'mean':>12}{'std':>12}")
    print("-" * 34)
    for m, v in ring.items():
        print(f"{m:<10}{v['mean']:>12.4f}{v['std']:>12.4f}")
    tag = ("tabular" if cfg.tanpa_sekuens
           else f"{cfg.representasi}_{cfg.encoder}_{cfg.agregasi}")
    json.dump({"mode": f"cv{cfg.cv}", "split": cfg.split, "tag": tag,
               "per_lipatan": kumpul, "ringkas": ring},
              open(cfg.dir_hasil / f"cv{cfg.cv}_{tag}_{cfg.split}.json", "w"),
              indent=2)
    print(f"\nDisimpan: cv{cfg.cv}_{tag}_{cfg.split}.json")
    return ring


def latih(cfg: Konfig, diam: bool = False) -> Dict:
    torch, nn, F = _torch()
    from torch.utils.data import TensorDataset, DataLoader

    d = np.load(cfg.dir_hasil / cfg.cache, allow_pickle=True)
    meta = json.load(open(cfg.dir_hasil / f"meta_{cfg.representasi}.json"))
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    y = d["y"].copy()
    if cfg.transformasi_target == "log1p":
        y = np.log1p(y).astype(np.float32)

    if getattr(cfg, "tekanan", None) is not None:
        target = float(np.log10(cfg.tekanan))
        pilih = np.abs(d["pres"][:, 0] - target) < 1e-6
        if pilih.sum() == 0:
            raise SystemExit(f"Tidak ada baris pada tekanan {cfg.tekanan} bar.")
        log.info("Filter tekanan %.2f bar -> %d dari %d baris",
                 cfg.tekanan, int(pilih.sum()), len(pilih))
        d = {k: (v[pilih] if hasattr(v, "__len__") and len(v) == len(pilih)
                 else v) for k, v in d.items()}
        y = y[pilih]

    lip = getattr(cfg, "_lipatan", None)
    if lip is not None:
        is_tr, is_va, is_te = lip
    else:
        is_tr, is_va, is_te = buat_split(cfg, d)

    # --- penskalaan fitur tabular numerik: statistik dari TRAIN saja ---
    TAB = d["tab"].copy()
    idx_num = meta.get("idx_tab_numerik")
    if idx_num is None:                      # dataset versi lama
        idx_num = [TAB.shape[1] - 1]
    if idx_num:
        mu = TAB[is_tr][:, idx_num].mean(0)
        sd = TAB[is_tr][:, idx_num].std(0) + 1e-8
        TAB[:, idx_num] = (TAB[:, idx_num] - mu) / sd
        log.info("Fitur tabular numerik diskalakan (%d kolom, "
                 "statistik dari train saja)", len(idx_num))
    if meta.get("kolom_tekstural"):
        log.info("Fitur tekstural dipakai: %s", meta["kolom_tekstural"])

    def loader(m, shuffle):
        ds = TensorDataset(torch.tensor(d["seq"][m]),
                           torch.tensor(d["lmask"][m]),
                           torch.tensor(TAB[m]),
                           torch.tensor(d["pres"][m]),
                           torch.tensor(y[m]))
        return DataLoader(ds, batch_size=cfg.batch, shuffle=shuffle)

    tr, va, te = loader(is_tr, True), loader(is_va, False), loader(is_te, False)
    tr_ev = loader(is_tr, False)

    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cpu"))
    model = bangun_model(cfg, meta["vocab_size"], meta["tab_dim"],
                         meta.get("maks_token")).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    if cfg.tanpa_sekuens:
        log.info("BASELINE TABULAR SAJA (string MOFHAPPY tidak dipakai) "
                 "| device=%s | parameter=%s", dev, f"{npar:,}")
    else:
        log.info("Encoder=%s | agregasi=%s | device=%s | parameter=%s",
                 cfg.encoder, cfg.agregasi, dev, f"{npar:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    fn = (F.mse_loss if cfg.loss == "mse"
          else lambda p, t: F.huber_loss(p, t, delta=1.0))

    def jalan(ld, o=None):
        model.train() if o else model.eval()
        P, T = [], []
        with torch.set_grad_enabled(o is not None):
            for s, lm, tb, pr, yy in ld:
                s, lm, tb, pr, yy = (s.to(dev), lm.to(dev), tb.to(dev),
                                     pr.to(dev), yy.to(dev))
                p = model(s, lm, tb, pr)
                loss = fn(p, yy)
                if o:
                    o.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    o.step()
                P.append(p.detach().cpu()); T.append(yy.cpu())
        return torch.cat(P).numpy(), torch.cat(T).numpy()

    tag = ("tabular" if cfg.tanpa_sekuens
           else f"{cfg.representasi}_{cfg.encoder}_{cfg.agregasi}")
    ckpt = cfg.dir_hasil / f"model_{tag}_{cfg.split}.pt"
    terbaik, sejak, ep_baik, riwayat = float("inf"), 0, 0, []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        jalan(tr, opt)
        pv, tv = jalan(va)
        mv = metrik(pv, tv)
        sch.step(mv["MSE"])
        riwayat.append({"epoch": ep, "val_MSE": mv["MSE"], "val_R2": mv["R2"]})
        if not diam and (ep == 1 or ep % 10 == 0):
            pt, tt = jalan(tr_ev)
            log.info("  epoch %3d | train R2 %.3f | val R2 %.3f MSE %.4f",
                     ep, metrik(pt, tt)["R2"], mv["R2"], mv["MSE"])
        if mv["MSE"] < terbaik - 1e-6:
            terbaik, sejak, ep_baik = mv["MSE"], 0, ep
            torch.save(model.state_dict(), ckpt)
        else:
            sejak += 1
            if sejak >= cfg.patience:
                log.info("  early stopping epoch %d (terbaik %d)", ep, ep_baik)
                break
    durasi = time.time() - t0

    model.load_state_dict(torch.load(ckpt))
    hasil = {}
    for nama, ld, msk in (("train", tr_ev, is_tr), ("val", va, is_va),
                          ("test", te, is_te)):
        p, t = jalan(ld)
        if cfg.transformasi_target == "log1p":
            p, t = np.expm1(p), np.expm1(t)      # metrik di skala asli
        hasil[nama] = metrik(p, t)
        if nama == "test":
            pd.DataFrame({"mof": d["mof"][msk], "y_asli": t,
                          "y_prediksi": p}).to_csv(
                cfg.dir_hasil / f"prediksi_{tag}_{cfg.split}.csv",
                index=False)

    if not diam:
        print(f"\n{'set':<8}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
        print("-" * 40)
        for k, v in hasil.items():
            print(f"{k:<8}{v['RMSE']:>10.4f}{v['MAE']:>10.4f}{v['R2']:>10.4f}")

    ringkas = {"representasi": cfg.representasi, "encoder": cfg.encoder,
               "agregasi": cfg.agregasi,
               "split": cfg.split,
               "loss": cfg.loss, "transformasi": cfg.transformasi_target,
               "n_parameter": int(npar), "epoch_terbaik": ep_baik,
               "detik_training": round(durasi, 1),
               "seed": cfg.seed, "hasil": hasil, "riwayat": riwayat}
    json.dump(ringkas, open(
        cfg.dir_hasil / f"ringkas_{tag}_{cfg.split}.json", "w"),
        indent=2)
    return ringkas


# ============================================================ TAHAP 4: ABLASI
def ablasi(cfg: Konfig) -> None:
    """Ablation study: apakah peningkatan berasal dari multi-linker,
    atau sekadar dari penambahan parameter?"""
    kombinasi = [
        # (encoder, agregasi, split) -- tiga kondisi untuk tiap encoder
        ("ann",  "concat_unshared", cfg.split),   # bobot BELUM disamakan
        ("rnn",  "concat_unshared", cfg.split),
        ("lstm", "concat_unshared", cfg.split),
        ("ann",  "concat_shared",   cfg.split),   # bobot SUDAH disamakan
        ("rnn",  "concat_shared",   cfg.split),
        ("lstm", "concat_shared",   cfg.split),
        ("ann",  "minmaxavg",       cfg.split),   # + agregasi min/max/avg
        ("rnn",  "minmaxavg",       cfg.split),
        ("lstm", "minmaxavg",       cfg.split),
    ]
    rekap = []
    for enc, agg, sp in kombinasi:
        log.info("=" * 60)
        log.info("ABLASI: encoder=%s agregasi=%s split=%s", enc, agg, sp)
        c = Konfig(**{**asdict(cfg), "encoder": enc,
                      "agregasi": agg, "split": sp})
        c.dir_data, c.dir_hasil = cfg.dir_data, cfg.dir_hasil
        r = latih(c, diam=True)
        t = r["hasil"]["test"]
        rekap.append({"encoder": enc, "agregasi": agg, "split": sp,
                      "parameter": r["n_parameter"],
                      "detik": r["detik_training"],
                      "test_R2": t["R2"], "test_RMSE": t["RMSE"],
                      "test_MAE": t["MAE"], "slope": t["slope"],
                      "val_R2": r["hasil"]["val"]["R2"],
                      "train_R2": r["hasil"]["train"]["R2"]})
    tab = pd.DataFrame(rekap)
    tab.to_csv(cfg.dir_hasil / "ablasi.csv", index=False)
    print("\n" + "=" * 96)
    print("HASIL ABLATION STUDY")
    print("=" * 96)
    print(tab.to_string(index=False))
    print("\nCara membaca:")
    print("  1. ann/rnn/lstm dgn agregasi 'single' = baseline single-linker.")
    print("  2. Encoder yang sama dgn 'attention'  = versi multi-linker.")
    print("     Selisih 1 vs 2 = kontribusi informasi multi-linker.")
    print("  3. Bandingkan 'single' vs 'mean'/'attention' pada split")
    print("yang SAMA. Selisihnya = kontribusi informasi multi-linker.")
    print("Baris terakhir (split=mof) menunjukkan berapa besar R2 melonjak")
    print("hanya karena split yang lebih longgar — itu bukan peningkatan nyata.")


# ============================================================ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Whole-MOF multi-linker pipeline")
    ap.add_argument("--tahap", required=True,
                    choices=["data", "diagnosa", "latih", "ablasi"])
    ap.add_argument("--dir-data", default=".")
    ap.add_argument("--dir-hasil", default="hasil")
    ap.add_argument("--berkas-linker", default="data_siap.csv")
    ap.add_argument("--berkas-label", default="hMOF_CO2_info.csv")
    ap.add_argument("--n-mof", type=int, default=20000)
    ap.add_argument("--tanpa-sekuens", action="store_true",
                    help="baseline: hanya fitur tabular + tekanan, "
                         "string MOFHAPPY diabaikan")
    ap.add_argument("--tekanan", type=float, default=None,
                    help="pakai satu tekanan saja, mis. 0.1 (seperti Hagel)")
    ap.add_argument("--tabular", default="",
                    help="berkas CSV/XLSX fitur tekstural (void fraction, "
                         "surface area, PLD, LCD); harus punya kolom 'mof'")
    ap.add_argument("--tabular-murni", action="store_true",
                    help="tabular HANYA fitur tekstural (setara Hagel)")
    ap.add_argument("--n-target", type=int, default=0,
                    help="batasi jumlah MOF, mis. 8423 seperti Hagel")
    ap.add_argument("--cv", type=int, default=0,
                    help="K-fold cross validation, mis. 5 (seperti Hagel)")
    ap.add_argument("--kolom-tabular", default="",
                    help="daftar kolom dipisah koma; kosong = semua numerik")
    ap.add_argument("--maks-linker", type=int, default=3,
                    help="jumlah linker dominan yang dipakai (sisanya dibuang)")
    ap.add_argument("--representasi", default="mofhappy",
                    choices=["mofhappy", "smiles"],
                    help="smiles = baseline tanpa coarse-graining")
    ap.add_argument("--encoder", default="gru",
                    choices=["ann", "rnn", "lstm", "gru"])
    ap.add_argument("--agregasi", default="minmaxavg",
                    choices=["single", "mean", "sum", "minmaxavg", "max",
                             "min", "attention", "deepsets",
                             "concat_unshared", "concat_shared"])
    ap.add_argument("--split", default="linkerset",
                    choices=["random", "mof", "linkerset", "linker"])
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--transformasi-target", default="none",
                    choices=["none", "log1p"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = Konfig(
        dir_data=Path(a.dir_data), dir_hasil=Path(a.dir_hasil),
        berkas_linker=a.berkas_linker, berkas_label=a.berkas_label,
        berkas_tabular=a.tabular,
        kolom_tabular=tuple(c.strip() for c in a.kolom_tabular.split(",")
                            if c.strip()),
        tabular_murni=a.tabular_murni, n_target=a.n_target, cv=a.cv,
        n_mof=a.n_mof if a.n_mof > 0 else None,
        maks_linker=a.maks_linker, representasi=a.representasi,
        encoder=a.encoder, tanpa_sekuens=a.tanpa_sekuens,
        tekanan=a.tekanan,
        agregasi=a.agregasi, split=a.split,
        loss=a.loss, transformasi_target=a.transformasi_target,
        epochs=a.epochs, seed=a.seed)
    cfg.dir_hasil.mkdir(parents=True, exist_ok=True)

    cfg.cache = f"wholemof_{cfg.representasi}.npz"
    if a.tahap == "data":
        bangun_dataset(cfg)
    elif a.tahap == "diagnosa":
        diagnosa(cfg)
    elif a.tahap == "latih":
        latih_cv(cfg) if cfg.cv else latih(cfg)
    else:
        ablasi(cfg)


if __name__ == "__main__":
    main()
