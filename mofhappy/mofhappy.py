"""
mofhappy.py -- MOFHAPPY v0.2 (tahap A)
A HAPPY-style coarse-grained string representation for MOFs (node + linker only).

Design goals (in priority order):
  1. Deterministic  : one MOF -> exactly one string.
  2. Ordered        : token order follows a canonical DFS traversal (unlike a naive
                      substructure-count encoder).
  3. Graceful       : unknown fragments degrade to a generic token, never a hard fail.
  4. Compact        : ~10-20x fewer tokens than MOFid.
  5. Partly invertible at the fragment-graph level.

String format:   H <NODE> ~ <LINKER> T
Example       :   H Zr6O ~ K-B-K T   (UiO-66)

Requires: rdkit  (pip install rdkit)
"""

from collections import Counter, defaultdict
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------------
# STEP 1 VOCABULARY
# ----------------------------------------------------------------------------

# 1a. Coordination groups (the "connector" concept of HAPPY, made chemical).
#     Order matters: more specific patterns first.
COORD_SMARTS = [
    ("Y", "[NX1]#[CX2]"),                           # nitril sebagai N-donor
    ("K", "[CX3](=O)[OX1H0-,OX2H1,OX1H0]"),        # carboxylate / carboxylic acid
    ("S", "[SX4](=O)(=O)[OX1H0-,OX2H1]"),          # sulfonate
    ("F", "[PX4](=O)([OX1H0-,OX2H1])[OX1H0-,OX2H1]"),  # phosphonate
    ("Z", "c1cnc[nH]1"),                            # imidazolate
    ("Z", "c1cn[nH]n1"),                            # tetrazolate
    ("W", "[nX2r6]1ccccc1"),                        # pyridyl N-donor
]

# 1b. Ring systems. Key = canonical SMILES of the *whole fused system*, H-suppressed.
#     Extend this dict from data (see STEP 4, build_ring_vocab).
RING_VOCAB = {
    "c1ccccc1": "B",            # benzene
    "c1ccc2ccccc2c1": "Bc",     # naphthalene
    "c1ccc2c(c1)ccc1ccccc12": "Ba",   # anthracene/phenanthrene family
    "c1cc2ccc3cccc4ccc(c1)c2c34": "Y",  # pyrene
    "C1CCCCC1": "Hx",           # cyclohexane
    "C1CCCC1": "Pn",            # cyclopentane
    "c1ccncc1": "N6",           # pyridine
    "c1cnccn1": "N2",           # pyrazine
    "c1ccc2[nH]cnc2c1": "Bn",   # benzimidazole
    "c1ccc2ccccc2c1": "Bc",
    "c1ccc2c(c1)C(=O)NC2=O": "Bh",   # phthalimide
    "O=C1C=CC(=O)N1": "Ma",     # maleimide
    "c1ccc2c(c1)[nH]c1ccccc12": "Nbb",  # carbazole
    "c1ccc(-c2ccccc2)cc1": "BB",       # biphenyl (kept as 2 x B by default)
    "c1cc2cc3ccc4cc5ccc1n5[nH]4c3[nH]2": "P4",  # porphyrin (approximate key)
}

# 1b-auto. Optional: extend RING_VOCAB from a CSV produced by step2 so you never
#          have to hand-edit this file. Columns: smiles_cincin,token

# Menambahkan pasangan SMILES dan token dari file CSV eksternal jika ada. Ini memungkinkan pengguna untuk memperluas kosakata ring secara dinamis tanpa harus mengubah kode sumber secara manual.
import os as _os, csv as _csv
_RING_CSV = _os.environ.get("MOFHAPPY_RING_CSV", "ring_vocab.csv")
if _os.path.exists(_RING_CSV):
    with open(_RING_CSV) as _f:
        for _row in _csv.DictReader(_f):
            _smi, _tok = _row.get("smiles_cincin"), _row.get("token")
            if _smi and _tok:
                RING_VOCAB.setdefault(_smi, _tok)

# v0.2: satu huruf tidak boleh punya dua arti. Token cincin yang sama dengan
# token gugus koordinasi (mis. pirena "Y" vs nitril "Y") diberi akhiran "r".
_COORD_TOKENS = {t for t, _ in COORD_SMARTS}
for _k, _v in list(RING_VOCAB.items()):
    if _v in _COORD_TOKENS:
        RING_VOCAB[_k] = _v + "r"

# 1c. Substituent modifiers (single characters inside braces).
SUBST_VOCAB = {
    "N": "A",     # -NH2
    "C": "M",     # -CH3
    "O": "O",     # -OH / -OR
    "F": "F",
    "Cl": "L",
    "Br": "R",
    "I": "I",
    "S": "T",
    "[N+](=O)[O-]": "X",   # nitro
}

# 1d. Metal node vocabulary. Built from composition, not from SMILES matching,
#     because MOF node SMILES have non-standard valence.
NODE_MOTIF_ORDER = ["O", "OH", "N", "C"]

GENERIC_RING = "X"        # unknown ring system
GENERIC_CHAIN = "C"       # aliphatic chain, written C<n>


# ----------------------------------------------------------------------------
# STEP 2 NODE ENCODER
# ----------------------------------------------------------------------------

def encode_node(node_smiles):
    """Turn a MOFid-style metal-node SMILES into a single MOFHAPPY token.

    Strategy: never parse valence. Just count metals + bridging light atoms.
        [Zn][O]([Zn])([Zn])[Zn]              -> Zn4O
        [Cu][Cu] with 4 carboxylate O        -> Cu2P   (paddlewheel, see below)
        Zr6 cluster with O/OH bridges        -> Zr6O
    """
    METALS = set("""Li Na K Rb Cs Be Mg Ca Sr Ba Sc Y Ti Zr Hf V Nb Ta Cr Mo W
        Mn Re Fe Ru Os Co Rh Ir Ni Pd Pt Cu Ag Au Zn Cd Hg Al Ga In Tl Sn Pb Bi
        La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Th U""".split())

    mol = Chem.MolFromSmiles(node_smiles, sanitize=False)
    if mol is None:
        return "M?"
    counts = Counter()
    for a in mol.GetAtoms():
        counts[a.GetSymbol()] += 1

    metals = [(s, n) for s, n in counts.items() if s in METALS]
    if not metals:
        return "M?"
    metals.sort(key=lambda x: (-x[1], x[0]))
    sym, n = metals[0]

    motif = ""
    if counts.get("O", 0) >= 1:
        motif = "O"
    if counts.get("N", 0) > counts.get("O", 0):
        motif = "N"

    token = f"{sym}{n}{motif}"

    # Well-known aliases: make the common cases human-readable.
    ALIAS = {
        "Cu2O": "Cu2P",   # paddlewheel
        "Zn2O": "Zn2P",
        "Zn4O": "Zn4O",
        "Zr6O": "Zr6O",
        "Hf6O": "Hf6O",
        "Cr3O": "Cr3O",
        "Fe3O": "Fe3O",
        "Al1O": "AlCh",   # Al-O chain (MIL-53 type)
    }
    return ALIAS.get(token, token)


# ----------------------------------------------------------------------------
# STEP 3 LINKER ENCODER
# ----------------------------------------------------------------------------

def _fused_ring_systems(mol):
    """Merge SSSR rings that share atoms -> list of atom-index sets."""
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        s = set(ring)
        merged = [s]
        for existing in systems:
            if existing & s:
                merged.append(existing)
        for m in merged[1:]:
            systems.remove(m)
        systems.append(set().union(*merged))
    return systems


def _ring_token(mol, atom_idxs):
    """Canonical SMILES of a ring system -> vocabulary token (or GENERIC_RING)."""
    try:
        smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atom_idxs),
                                       canonical=True)
        smi = Chem.CanonSmiles(smi)
    except Exception:
        return GENERIC_RING, None
    return RING_VOCAB.get(smi, GENERIC_RING), smi


def _ring_pattern(mol, atom_idxs, attach_atoms):
    """For a 6-membered aromatic ring with exactly 2 attachments, return the
    substitution pattern suffix: '' = para, 'm' = meta, 'o' = ortho.
    This is what makes the representation partly invertible."""
    if len(atom_idxs) != 6 or len(attach_atoms) != 2:
        return ""
    a, b = list(attach_atoms)
    path = Chem.GetShortestPath(mol, a, b)
    d = len(path) - 1
    return {1: "o", 2: "m", 3: ""}.get(d, "")


def _chain_token(mol, comp):
    """v0.2: komposisi unsur (dengan jumlah) + jumlah ikatan rangkap dua (=n)
    dan rangkap tiga (#n) di dalam rantai. Awalan "_" membedakan token rantai
    dari token gugus koordinasi dan token cincin (mis. F, S, Y).
        C#C-C(Cl)=C(Cl)-C#C -> _C6Cl2=1#2        (bukan lagi "CCl")
    """
    cnt = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in comp)
    bagian = []
    for sym in sorted(cnt, key=lambda x: (x != "C", x)):
        n = cnt[sym]
        bagian.append(sym if n == 1 else f"{sym}{n}")
    himp = set(comp)
    n_dua = n_tiga = 0
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in himp and b.GetEndAtomIdx() in himp:
            bt = b.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                n_dua += 1
            elif bt == Chem.BondType.TRIPLE:
                n_tiga += 1
    out = "_" + "".join(bagian)
    if n_dua:
        out += f"={n_dua}"
    if n_tiga:
        out += f"#{n_tiga}"
    return out


def fragment_linker(smiles):
    """Partition a linker into fragments. Returns (mol, frags) where each frag is
    a dict: {kind, atoms, token}."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.RemoveHs(mol)

    assigned = {}          # atom idx -> frag id
    frags = []

    def new_frag(kind, atoms, token):
        fid = len(frags)
        frags.append({"kind": kind, "atoms": set(atoms), "token": token})
        for i in atoms:
            assigned[i] = fid
        return fid

    # 3a. Coordination groups first (highest priority).
    #     Matches are visited in CANONICAL order, not RDKit's internal atom
    #     order. Without this, two overlapping coordination patterns can be
    #     resolved differently depending on how the SMILES happened to be
    #     written, breaking determinism.
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    for token, sma in COORD_SMARTS:
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            continue
        matches = sorted(mol.GetSubstructMatches(patt),
                         key=lambda m: sorted(rank[i] for i in m))
        for match in matches:
            if any(i in assigned for i in match):
                continue
            new_frag("coord", match, token)

    # 3b. Ring systems.
    for sysatoms in _fused_ring_systems(mol):
        free = [i for i in sysatoms if i not in assigned]
        if not free:
            continue
        token, _ = _ring_token(mol, sysatoms)
        new_frag("ring", free, token)

    # 3c. Remaining atoms -> chain / heteroatom fragments (connected components).
    remaining = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in assigned]
    seen = set()
    for start in remaining:
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            i = stack.pop()
            if i in seen or i in assigned:
                continue
            seen.add(i)
            comp.append(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() not in assigned and nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
        token = _chain_token(mol, comp)          # v0.2
        new_frag("chain", comp, token)

    # 3d. Fold small terminal groups into {modifier} suffixes on their parent
    #     fragment. This is the step that gives HAPPY-like compactness.
    assigned2 = {}
    for fid, f in enumerate(frags):
        for i in f["atoms"]:
            assigned2[i] = fid
    nbr = defaultdict(set)
    for b in mol.GetBonds():
        fi, fj = assigned2.get(b.GetBeginAtomIdx()), assigned2.get(b.GetEndAtomIdx())
        if fi is not None and fj is not None and fi != fj:
            nbr[fi].add(fj); nbr[fj].add(fi)
    folded = set()
    for fid, f in enumerate(frags):
        if f["kind"] != "chain" or len(nbr[fid]) != 1 or len(f["atoms"]) > 1:
            continue
        sym = mol.GetAtomWithIdx(next(iter(f["atoms"]))).GetSymbol()
        mod = SUBST_VOCAB.get(sym)
        if mod is None:
            continue
        parent = next(iter(nbr[fid]))
        if frags[parent]["kind"] != "ring":
            continue
        frags[parent].setdefault("mods", []).append(mod)
        folded.add(fid)
    if folded:
        keep = [f for i, f in enumerate(frags) if i not in folded]
        frags = keep

    # 3e. Attach substitution-pattern suffixes to rings.
    assigned = {}
    for fid, f in enumerate(frags):
        for i in f["atoms"]:
            assigned[i] = fid
    bonds_between = defaultdict(set)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        fi, fj = assigned.get(i), assigned.get(j)
        if fi is not None and fj is not None and fi != fj:
            bonds_between[fi].add(i)
            bonds_between[fj].add(j)
    for fid, f in enumerate(frags):
        if f["kind"] == "ring" and f["token"] == "B":
            f["token"] += _ring_pattern(mol, f["atoms"], bonds_between[fid])
        if f.get("mods"):
            f["token"] += "{" + "".join(sorted(f["mods"])) + "}"

    return mol, frags


def _frag_graph(mol, frags):
    assigned = {}
    for fid, f in enumerate(frags):
        for i in f["atoms"]:
            assigned[i] = fid
    adj = defaultdict(set)
    for b in mol.GetBonds():
        fi = assigned.get(b.GetBeginAtomIdx())
        fj = assigned.get(b.GetEndAtomIdx())
        if fi is not None and fj is not None and fi != fj:
            adj[fi].add(fj)
            adj[fj].add(fi)
    return adj


def _serialize(fid, parent, frags, adj, visited=None):
    """Canonical DFS over the fragment graph.

    Branches are sorted by their own serialized string, so the result does not
    depend on RDKit atom ordering. The fragment graph may contain cycles (e.g.
    macrocyclic or multiply-bridged linkers), so a visited set is required;
    a back edge is recorded with a '%' ring-closure marker on the token
    instead of being silently dropped.
    """
    if visited is None:
        visited = set()
    visited.add(fid)
    tok = frags[fid]["token"]

    children, closures = [], 0
    for n in sorted(adj[fid]):
        if n == parent:
            continue
        if n in visited:
            closures += 1
        else:
            children.append(n)

    subs = sorted(_serialize(c, fid, frags, adj, visited)
                  for c in children if c not in visited or True)
    if closures:
        tok += "%" * closures
    if not subs:
        return tok
    if len(subs) == 1:
        return f"{tok}-{subs[0]}"
    return tok + "".join(f"({s})" for s in subs[:-1]) + f"-{subs[-1]}"


def encode_linker(smiles):
    mol, frags = fragment_linker(smiles)
    if mol is None:
        return None
    adj = _frag_graph(mol, frags)

    # Root = a coordination fragment if present (chemically meaningful anchor),
    # else the fragment whose serialization sorts first.
    coord_ids = [i for i, f in enumerate(frags) if f["kind"] == "coord"]
    candidates = coord_ids if coord_ids else list(range(len(frags)))
    best = min(_serialize(c, None, frags, adj, set()) for c in candidates)
    return best


def encode_mof(node_smiles, linker_smiles):
    return f"H {encode_node(node_smiles)} ~ {encode_linker(linker_smiles)} T"


# ----------------------------------------------------------------------------
# STEP 4 VOCABULARY BUILDING FROM DATA
# ----------------------------------------------------------------------------

def build_ring_vocab(linker_smiles_list, top_n=60):
    """Run this on your CoRE MOF linker list FIRST. It returns the ring systems
    ranked by frequency so you can decide where to cut the vocabulary."""
    c = Counter()
    for smi in linker_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for sysatoms in _fused_ring_systems(mol):
            _, canon = _ring_token(mol, sysatoms)
            if canon:
                c[canon] += 1
    return c.most_common(top_n)


def build_node_vocab(node_smiles_list, top_n=60):
    c = Counter(encode_node(s) for s in node_smiles_list)
    return c.most_common(top_n)


# ----------------------------------------------------------------------------
# STEP 5 EVALUATION METRICS
# ----------------------------------------------------------------------------

def tokenize(s):
    """Split a MOFHAPPY string into tokens for length statistics."""
    out, buf = [], ""
    for ch in s:
        if ch in "-()~ ":
            if buf:
                out.append(buf); buf = ""
            if ch not in " ":
                out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def evaluate(pairs):
    """pairs = list of (node_smiles, linker_smiles). Reports the three numbers
    that actually matter for your paper."""
    n_ok = n_generic = 0
    happy_len, smiles_len = [], []
    for node, link in pairs:
        s = encode_mof(node, link)
        if s is None or "None" in s:
            continue
        n_ok += 1
        toks = tokenize(s)
        if GENERIC_RING in toks:
            n_generic += 1
        happy_len.append(len(toks))
        smiles_len.append(len(node) + len(link))
    if not n_ok:
        return {}
    return {
        "n_encoded": n_ok,
        "coverage_no_unknown_%": round(100 * (n_ok - n_generic) / n_ok, 1),
        "mean_tokens_MOFHAPPY": round(sum(happy_len) / len(happy_len), 2),
        "mean_chars_SMILES": round(sum(smiles_len) / len(smiles_len), 2),
        "compression_x": round((sum(smiles_len) / len(smiles_len)) /
                               (sum(happy_len) / len(happy_len)), 1),
    }


# ----------------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    EXAMPLES = [
        ("MOF-5",    "[Zn][O]([Zn])([Zn])[Zn]", "O=C([O-])c1ccc(C(=O)[O-])cc1"),
        ("UiO-66",   "[Zr][O]([Zr])([Zr])([Zr])([Zr])[Zr]",
                     "O=C([O-])c1ccc(C(=O)[O-])cc1"),
        ("UiO-67",   "[Zr][O]([Zr])([Zr])([Zr])([Zr])[Zr]",
                     "O=C([O-])c1ccc(-c2ccc(C(=O)[O-])cc2)cc1"),
        ("UiO-66-NH2", "[Zr][O]([Zr])([Zr])([Zr])([Zr])[Zr]",
                     "Nc1cc(C(=O)[O-])ccc1C(=O)[O-]"),
        ("HKUST-1",  "[Cu][O][Cu]", "O=C([O-])c1cc(C(=O)[O-])cc(C(=O)[O-])c1"),
        ("ZIF-8",    "[Zn]", "Cc1ncc[nH]1"),
        ("MIL-101",  "[Cr][O]([Cr])[Cr]", "O=C([O-])c1ccc(C(=O)[O-])cc1"),
        ("MOF-74",   "[Zn][O][Zn]", "O=C([O-])c1ccc(O)c(C(=O)[O-])c1O"),
        ("Cu-BTB",   "[Cu][O][Cu]",
                     "O=C([O-])c1ccc(-c2cc(-c3ccc(C(=O)[O-])cc3)cc(-c3ccc(C(=O)[O-])cc3)c2)cc1"),
    ]

    print("=" * 78)
    print(f"{'MOF':<14}{'MOFHAPPY':<34}{'tokens':>7}{'SMILES chars':>14}")
    print("=" * 78)
    pairs = []
    for name, node, link in EXAMPLES:
        s = encode_mof(node, link)
        pairs.append((node, link))
        print(f"{name:<14}{s:<34}{len(tokenize(s)):>7}{len(node)+len(link):>14}")

    print("\nMetrics:")
    for k, v in evaluate(pairs).items():
        print(f"  {k:<26} {v}")

    print("\nRing-vocabulary frequency scan (run this on your real linker list):")
    for canon, n in build_ring_vocab([l for _, _, l in EXAMPLES], top_n=8):
        print(f"  {n:>3}x  {canon:<45} -> {RING_VOCAB.get(canon, GENERIC_RING)}")
