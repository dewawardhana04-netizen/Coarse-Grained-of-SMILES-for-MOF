"""uji_permutasi.py — Bukti empiris permutation-invariance.

Ambil sebuah MOF, acak urutan slot linkernya beberapa kali,
lalu bandingkan keluaran model. Jika invariant, semua keluaran identik.
"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, ".")
from wholemof_pipeline import Konfig, bangun_model

def cari_berkas(dir_hasil=None):
    """Temukan pasangan .npz + meta .json apa pun versi pipeline-nya.

    Versi lama menulis  wholemof_data.npz  + meta_data.json
    Versi baru menulis  wholemof_<repr>.npz + meta_<repr>.json
    """
    kandidat_dir = ([Path(dir_hasil)] if dir_hasil else
                    sorted(Path(".").glob("hasil*")) + [Path(".")])
    for dh in kandidat_dir:
        if not dh.is_dir():
            continue
        for npz in sorted(dh.glob("wholemof_*.npz")):
            suf = npz.stem.replace("wholemof_", "")
            for meta in (dh / f"meta_{suf}.json", dh / "meta_data.json"):
                if meta.exists():
                    return npz, meta
    raise SystemExit(
        "Berkas dataset tidak ditemukan.\n"
        "Jalankan dulu:  python3 wholemof_pipeline.py --tahap data --n-mof 2000\n"
        f"Dicari di: {[str(x) for x in kandidat_dir]}")


def uji(dir_hasil=None, agregasi="attention", n_uji=200, n_acak=8,
        _cache={}):
    if "f" not in _cache:
        _cache["f"] = cari_berkas(dir_hasil)
        print(f"Dataset : {_cache['f'][0]}")
    npz, meta_p = _cache["f"]
    d = np.load(npz, allow_pickle=True)
    m = json.load(open(meta_p))
    cfg = Konfig(agregasi=agregasi, encoder="ann")
    torch.manual_seed(0)
    # bangun_model() versi lama menerima 3 argumen, versi baru 4.
    # Deteksi otomatis supaya skrip ini bekerja pada kedua versi pipeline.
    import inspect
    n_arg = len(inspect.signature(bangun_model).parameters)
    if n_arg >= 4:
        model = bangun_model(cfg, m["vocab_size"], m["tab_dim"],
                             m.get("maks_token", d["seq"].shape[2]))
    else:
        cfg.maks_token = m.get("maks_token", d["seq"].shape[2])
        model = bangun_model(cfg, m["vocab_size"], m["tab_dim"])
    model.eval()

    # hanya MOF dengan >=2 linker (kalau 1 linker, permutasi tak bermakna)
    idx = np.where(d["lmask"].sum(1) >= 2)[0][:n_uji]
    seq = torch.tensor(d["seq"][idx]); lm = torch.tensor(d["lmask"][idx])
    tb = torch.tensor(d["tab"][idx]); pr = torch.tensor(d["pres"][idx])

    with torch.no_grad():
        acuan = model(seq, lm, tb, pr).numpy()
        maks_beda = 0.0
        for _ in range(n_acak):
            p = torch.stack([torch.randperm(seq.shape[1])
                             for _ in range(len(idx))])
            s2 = torch.gather(seq, 1, p.unsqueeze(-1).expand(-1, -1, seq.shape[2]))
            l2 = torch.gather(lm, 1, p)
            out = model(s2, l2, tb, pr).numpy()
            maks_beda = max(maks_beda, float(np.abs(out - acuan).max()))
    print(f"agregasi={agregasi:<10} MOF diuji={len(idx)}  "
          f"permutasi={n_acak}  selisih maksimum = {maks_beda:.3e}  "
          f"-> {'INVARIANT' if maks_beda < 1e-4 else 'VARIANT (bergantung urutan)'}")

if __name__ == "__main__":
    dh = sys.argv[1] if len(sys.argv) > 1 else None
    for a in ["concat_unshared", "concat_shared", "single",
              "mean", "minmaxavg", "attention"]:
        uji(dir_hasil=dh, agregasi=a)
