#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CA_PDB = ROOT / "data" / "ca_pdb"
RESULTS.mkdir(parents=True, exist_ok=True)

TMALIGN = (
    os.environ.get("TMALIGN")
    or shutil.which("TMalign")
    or str(Path.home() / "Desktop/esmc_trypsin_structure/bin/TMalign")
)
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))

AAS = "ACDEFGHIKLMNPQRSTVWY"

_TM_RE = re.compile(r"TM-score=\s*([\d.]+)")
_RMSD_RE = re.compile(r"RMSD=\s*([\d.]+)")
_ALN_RE = re.compile(r"Aligned length=\s*(\d+)")
_SEQID_RE = re.compile(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)")


def tm_align_pair(args: tuple[str, str]) -> dict:
    id_a, id_b = args
    pa, pb = CA_PDB / f"{id_a}.pdb", CA_PDB / f"{id_b}.pdb"
    blank = {"tm_score": np.nan, "tm_min": np.nan, "rmsd": np.nan,
             "n_aligned": np.nan, "pid_struct_aln": np.nan}
    try:
        r = subprocess.run([TMALIGN, str(pa), str(pb)],
                           capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return blank
    tms = [float(x) for x in _TM_RE.findall(r.stdout)]
    if len(tms) < 2:
        return blank
    rmsd = _RMSD_RE.search(r.stdout)
    aln = _ALN_RE.search(r.stdout)
    sid = _SEQID_RE.search(r.stdout)
    return {
        "tm_score": float(np.mean(tms[:2])),
        "tm_min": float(min(tms[:2])),
        "rmsd": float(rmsd.group(1)) if rmsd else np.nan,
        "n_aligned": int(aln.group(1)) if aln else np.nan,
        "pid_struct_aln": float(sid.group(1)) if sid else np.nan,
    }


def tm_d0(length: int) -> float:
    length = max(length, 16)
    return 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8


def kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    v, _, wt = np.linalg.svd(p_c.T @ q_c)
    if np.linalg.det(v @ wt) < 0:
        v[:, -1] *= -1
    return p_c @ v @ wt


def seqdep_scores(seq_a, ca_a, seq_b, ca_b, alignment) -> dict:
    xyz_a = np.array([xyz for _, xyz in ca_a], dtype=float)
    xyz_b = np.array([xyz for _, xyz in ca_b], dtype=float)
    if len(xyz_a) != len(seq_a) or len(xyz_b) != len(seq_b):
        n = min(len(xyz_a), len(seq_a))
        xyz_a, seq_a = xyz_a[:n], seq_a[:n]
        n = min(len(xyz_b), len(seq_b))
        xyz_b, seq_b = xyz_b[:n], seq_b[:n]

    a_aln, b_aln = alignment[0], alignment[1]
    ia = ib = 0
    pa, pb = [], []
    matches = aligned_cols = 0
    for ca, cb in zip(a_aln, b_aln):
        if ca != "-":
            ia_now = ia
            ia += 1
        if cb != "-":
            ib_now = ib
            ib += 1
        if ca != "-" and cb != "-":
            aligned_cols += 1
            if ca == cb:
                matches += 1
            pa.append(xyz_a[ia_now])
            pb.append(xyz_b[ib_now])

    lref = min(len(seq_a), len(seq_b))
    out = {
        "pid_aligned": matches / aligned_cols if aligned_cols else np.nan,
        "pid_short": matches / lref if lref else np.nan,
        "n_aligned_seq": int(len(pa)),
    }
    if len(pa) < 30:
        out.update({"tm_seqdep": np.nan, "rmsd_seqdep": np.nan})
        return out
    p, q = np.asarray(pa), np.asarray(pb)
    d = np.linalg.norm(kabsch(p, q) - (q - q.mean(axis=0)), axis=1)
    d0 = tm_d0(lref)
    out.update({
        "rmsd_seqdep": float(np.sqrt(np.mean(d ** 2))),
        "tm_seqdep": float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / lref),
    })
    return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


def composition(seq: str) -> np.ndarray:
    v = np.array([seq.count(a) for a in AAS], dtype=float)
    return v / max(v.sum(), 1.0)


def main() -> None:
    meta = pd.read_csv(ROOT / "data" / "proteins.csv")
    coords = json.loads((ROOT / "data" / "ca_coords.json").read_text())
    emb = np.load(RESULTS / "esmc300m_meanpool.npy")
    ids = [ln.strip() for ln in
           (RESULTS / "protein_ids.txt").read_text().splitlines() if ln.strip()]
    if ids != meta["id"].tolist():
        raise SystemExit(
            "Embedding ID order does not match proteins.csv.\n"
            "The cohort changed since 02_embed_esmc.py last ran. Re-run it.")

    if not Path(TMALIGN).exists():
        raise SystemExit(
            f"TM-align not found at {TMALIGN}\n"
            "Set $TMALIGN, or compile it:\n"
            "  g++ -O3 -ffast-math -o TMalign TMalign.cpp\n"
            "This script will NOT silently fall back to the sequence-dependent\n"
            "score -- that is the defect it exists to remove.")
    missing = [i for i in meta["id"] if not (CA_PDB / f"{i}.pdb").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} CA PDBs missing from {CA_PDB} (e.g. {missing[:3]}).\n"
            "Re-run 01_fetch_dataset.py, then scripts/01b_verify_views.py.")

    n = len(meta)
    pairs = list(combinations(range(n), 2))
    print(f"{n} proteins, {len(pairs)} pairs")

    print(f"TM-align on {N_WORKERS} workers ...", flush=True)
    jobs = [(meta.iloc[i]["id"], meta.iloc[j]["id"]) for i, j in pairs]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        tm_results = list(ex.map(tm_align_pair, jobs, chunksize=32))
    n_fail = sum(1 for t in tm_results if not np.isfinite(t["tm_score"]))
    if n_fail:
        print(f"  ! {n_fail} TM-align pairs failed to parse")
    print(f"  done, {len(tm_results) - n_fail} pairs scored")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    comps = {r["id"]: composition(r["sequence"]) for _, r in meta.iterrows()}

    rows = []
    for k, ((i, j), tm) in enumerate(zip(pairs, tm_results)):
        a, b = meta.iloc[i], meta.iloc[j]
        aln = aligner.align(a["sequence"], b["sequence"])[0]
        sd = seqdep_scores(a["sequence"], coords[a["id"]],
                           b["sequence"], coords[b["id"]], aln)
        rows.append({
            "id_a": a["id"], "id_b": b["id"],
            "fold_a": a["fold"], "fold_b": b["fold"],
            "subclass_a": a["subclass"], "subclass_b": b["subclass"],
            "kingdom_a": a["kingdom"], "kingdom_b": b["kingdom"],
            "same_fold": int(a["fold"] == b["fold"]),
            "same_subclass": int(a["subclass"] == b["subclass"]),
            "same_kingdom": int(a["kingdom"] == b["kingdom"]),
            "same_superkingdom": int(a.get("superkingdom") == b.get("superkingdom")),
            "emb_cos": cosine_distance(emb[i], emb[j]),
            "comp_dist": float(np.linalg.norm(comps[a["id"]] - comps[b["id"]])),
            "len_diff": abs(int(a["length"]) - int(b["length"])),
            "cys_diff": abs(int(a["cys_count"]) - int(b["cys_count"])),
            **tm,
            **sd,
        })
        if (k + 1) % 500 == 0:
            print(f"  pairs {k + 1}/{len(pairs)}", flush=True)

    df = pd.DataFrame(rows)
    df["seq_dist"] = 1.0 - df["pid_short"]
    df["str_dist"] = 1.0 - df["tm_score"]
    df["str_dist_seqdep"] = 1.0 - df["tm_seqdep"]
    out = RESULTS / "pairwise.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} pairs -> {out}")

    sub = df[(df.same_fold == 1) & df.tm_score.notna() & df.tm_seqdep.notna()]
    if len(sub):
        print("\n-- old (sequence-dependent) vs new (TM-align), same-fold pairs --")
        bins = pd.cut(sub["pid_short"],
                      [0, .25, .35, .45, .60, 1.01],
                      labels=["<25%", "25-35%", "35-45%", "45-60%", ">60%"])
        tab = sub.groupby(bins, observed=True).agg(
            n=("tm_score", "size"),
            tm_align=("tm_score", "median"),
            tm_seqdep=("tm_seqdep", "median"),
        )
        tab["bias"] = (tab["tm_seqdep"] - tab["tm_align"]).round(3)
        print(tab.round(3).to_string())
        lo = sub[sub.pid_short < 0.30]
        print(f"\n  same-fold pairs scored below TM 0.50:")
        print(f"    TM-align              {(sub.tm_score < .5).mean() * 100:5.1f}%")
        print(f"    sequence-dependent    {(sub.tm_seqdep < .5).mean() * 100:5.1f}%")
        if len(lo):
            print(f"  below 30% identity (n={len(lo)}), median TM:")
            print(f"    TM-align              {lo.tm_score.median():.3f}")
            print(f"    sequence-dependent    {lo.tm_seqdep.median():.3f}")
        r = sub[["tm_score", "tm_seqdep"]].corr(method="spearman").iloc[0, 1]
        print(f"\n  Spearman(TM-align, sequence-dependent) = {r:.3f}")


if __name__ == "__main__":
    main()
