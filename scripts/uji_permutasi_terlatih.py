"""
uji_permutasi_terlatih.py
=============================================================================
Bukti permutation-invariance pada model yang SUDAH DILATIH.

Bedanya dengan uji_permutasi.py:
    uji_permutasi.py          -> model berbobot acak (menguji arsitektur)
    uji_permutasi_terlatih.py -> model hasil training (menguji model nyata)

Yang diuji: seluruh test set di-prediksi dua kali,
    (a) dengan urutan linker asli
    (b) dengan urutan linker diacak beberapa kali
lalu dibandingkan prediksinya DAN metrik akhirnya (RMSE/MAE/R2).

Model permutation-invariant akan menghasilkan prediksi yang identik dan
metrik yang identik sampai digit terakhir. Model permutation-variant akan
berubah.

Pakai:
    python3 uji_permutasi_terlatih.py                 # semua checkpoint
    python3 uji_permutasi_terlatih.py hasil           # folder tertentu
=============================================================================
"""
import sys
import json
import re
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from wholemof_pipeline import Konfig, bangun_model, buat_split, metrik  # noqa


def cari_data(dirh: Path):
    for npz in sorted(dirh.glob("wholemof_*.npz")):
        suf = npz.stem.replace("wholemof_", "")
        for meta in (dirh / f"meta_{suf}.json", dirh / "meta_data.json"):
            if meta.exists():
                return npz, meta
    raise SystemExit(f"Dataset tidak ditemukan di {dirh}. "
                     f"Jalankan dulu: --tahap data")


def urai_nama(stem: str):
    """model_[repr_]<encoder>_<agregasi>_<split> -> (repr, enc, agg, split)"""
    bag = stem.replace("model_", "").split("_")
    if len(bag) < 3:
        return None
    REPR = {"mofhappy", "smiles"}
    ENC = {"ann", "rnn", "lstm", "gru"}
    rep = bag[0] if bag[0] in REPR else "mofhappy"
    sisa = bag[1:] if bag[0] in REPR else bag
    if sisa[0] not in ENC:
        return None
    return rep, sisa[0], "_".join(sisa[1:-1]), sisa[-1]


def main(dirh="hasil", n_acak=5, seed=0):
    dirh = Path(dirh)
    npz, meta_p = cari_data(dirh)
    d = np.load(npz, allow_pickle=True)
    m = json.load(open(meta_p))
    print(f"Dataset : {npz}")

    ckpts = sorted(dirh.glob("model_*.pt"))
    if not ckpts:
        raise SystemExit(f"Tidak ada checkpoint model_*.pt di {dirh}.")

    dev = torch.device("cpu")
    rng = np.random.default_rng(seed)
    hasil = []

    print(f"\n{'model':<26}{'selisih maks':>14}{'ΔR2':>12}"
          f"{'ΔRMSE':>12}  status")
    print("-" * 78)

    for ck in ckpts:
        info = urai_nama(ck.stem)
        if info is None:
            continue
        rep, enc, agg, sp = info
        cfg = Konfig(encoder=enc, agregasi=agg, split=sp,
                     representasi=rep,
                     maks_linker=int(d["lmask"].shape[1]),
                     maks_token=int(d["seq"].shape[2]))
        try:
            model = bangun_model(cfg, m["vocab_size"], m["tab_dim"],
                                 m.get("maks_token", d["seq"].shape[2]))
            model.load_state_dict(torch.load(ck, map_location=dev))
        except Exception as e:                     # arsitektur tak cocok
            print(f"{enc+'-'+agg:<26}  (dilewati: {type(e).__name__})")
            continue
        model.eval()

        # ---- ambil test set sesuai split yang dipakai saat training ----
        _, _, is_te = buat_split(cfg, d)
        seq = torch.tensor(d["seq"][is_te])
        lm = torch.tensor(d["lmask"][is_te])
        tb = torch.tensor(d["tab"][is_te])
        pr = torch.tensor(d["pres"][is_te])
        y = d["y"][is_te]

        with torch.no_grad():
            acuan = model(seq, lm, tb, pr).numpy()
            m_acuan = metrik(acuan, y)

            beda_maks, dr2, drmse = 0.0, 0.0, 0.0
            for _ in range(n_acak):
                p = torch.stack([torch.randperm(seq.shape[1])
                                 for _ in range(len(seq))])
                s2 = torch.gather(
                    seq, 1, p.unsqueeze(-1).expand(-1, -1, seq.shape[2]))
                l2 = torch.gather(lm, 1, p)
                out = model(s2, l2, tb, pr).numpy()
                mo = metrik(out, y)
                beda_maks = max(beda_maks, float(np.abs(out - acuan).max()))
                dr2 = max(dr2, abs(mo["R2"] - m_acuan["R2"]))
                drmse = max(drmse, abs(mo["RMSE"] - m_acuan["RMSE"]))

        status = "INVARIANT" if beda_maks < 1e-4 else "VARIANT"
        print(f"{enc.upper()+'-'+agg:<26}{beda_maks:>14.3e}{dr2:>12.2e}"
              f"{drmse:>12.2e}  {status}")
        hasil.append({"encoder": enc, "agregasi": agg, "split": sp,
                      "R2_asli": round(m_acuan["R2"], 6),
                      "RMSE_asli": round(m_acuan["RMSE"], 6),
                      "selisih_prediksi_maks": beda_maks,
                      "delta_R2": dr2, "delta_RMSE": drmse,
                      "status": status})

    if hasil:
        import pandas as pd
        pd.DataFrame(hasil).to_csv("uji_permutasi_terlatih.csv", index=False)
        print("\nDisimpan: uji_permutasi_terlatih.csv")
        print(f"Setiap model diuji dengan {n_acak} pengacakan urutan linker "
              f"pada seluruh test set.")
        print("Selisih di orde 1e-7 atau lebih kecil = presisi floating-point,")
        print("bukan ketergantungan urutan.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hasil")
