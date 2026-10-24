#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests
from Bio.Data.IUPACData import protein_letters_1to3, protein_letters_3to1_extended
from Bio.PDB import MMCIFParser, PDBParser

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
STRUCT = DATA / "structures"
CA_PDB = DATA / "ca_pdb"
for d in (RAW, STRUCT, CA_PDB):
    d.mkdir(parents=True, exist_ok=True)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "trypsin-esmc-structure/0.2"})

AA3 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
AA1TO3 = {k.upper(): v.upper() for k, v in protein_letters_1to3.items()}

MIN_OBS_LEN = 180
MAX_OBS_LEN = 400
MIN_CA = 180
MAX_GAP = 12
MAX_MISSING = 30
MIN_COVERAGE = 0.90
MIN_FOLD_TM = 0.50
PANEL_SIZE = 5
ORG_CAP = 10

TMALIGN = (
    os.environ.get("TMALIGN")
    or shutil.which("TMalign")
    or str(Path.home() / "Desktop/esmc_trypsin_structure/bin/TMalign")
)


class ChainNotFound(Exception):
    pass


QUERIES = {
    "S1_trypsin": (
        '(xref:pfam-PF00089) AND (reviewed:true) AND (database:pdb) '
        'AND (length:[180 TO 600])'
    ),
    "S8_subtilisin": (
        '(xref:pfam-PF00082) AND (reviewed:true) AND (database:pdb) '
        'AND (length:[200 TO 1200])'
    ),
}

SUPPLEMENTARY_QUERIES = {
    "S8_subtilisin": [
        '(xref:pfam-PF00082) AND (reviewed:true) AND (database:pdb) '
        'AND (length:[200 TO 1200]) AND (taxonomy_id:33208)',
        '(xref:pfam-PF00082) AND (reviewed:true) AND (database:pdb) '
        'AND (length:[200 TO 1200]) AND (taxonomy_id:33090)',
    ],
    "S1_trypsin": [
        '(xref:pfam-PF00089) AND (reviewed:true) AND (database:pdb) '
        'AND (length:[180 TO 600]) AND (taxonomy_id:2)',
        '(xref:pfam-PF00089) AND (reviewed:true) AND (database:pdb) '
        'AND (length:[180 TO 600]) AND (taxonomy_id:4751)',
    ],
}

EC_SUBCLASS = [
    ("trypsin", ("3.4.21.4",)),
    ("chymotrypsin", ("3.4.21.1", "3.4.21.2")),
    ("elastase", ("3.4.21.36", "3.4.21.37", "3.4.21.71")),
    ("thrombin", ("3.4.21.5",)),
    ("kallikrein", ("3.4.21.34", "3.4.21.35", "3.4.21.118", "3.4.21.119")),
    ("coagulation_factor", ("3.4.21.6", "3.4.21.21", "3.4.21.22", "3.4.21.27")),
    ("complement", ("3.4.21.41", "3.4.21.42", "3.4.21.43", "3.4.21.45", "3.4.21.46", "3.4.21.47")),
    ("granzyme", ("3.4.21.78", "3.4.21.79")),
    ("subtilisin", ("3.4.21.62", "3.4.21.14", "3.4.21.64")),
    ("proprotein_convertase", ("3.4.21.75", "3.4.21.93", "3.4.21.94")),
    ("tripeptidyl_peptidase", ("3.4.14.9",)),
]


def classify_subclass(ec_list: list[str], name: str, family: str) -> str:
    name_l = name.lower()
    ecs = set(ec_list)
    for label, codes in EC_SUBCLASS:
        if any(code in ecs for code in codes):
            return label
    keywords = [
        ("site1_protease", "site-1"),
        ("proprotein_convertase", "convertase"),
        ("proprotein_convertase", "furin"),
        ("tripeptidyl_peptidase", "tripeptidyl"),
        ("subtilase", "subtilase"),
        ("cucumisin", "cucumisin"),
        ("trypsin", "trypsin"),
        ("chymotrypsin", "chymotrypsin"),
        ("elastase", "elastase"),
        ("thrombin", "thrombin"),
        ("kallikrein", "kallikrein"),
        ("coagulation_factor", "factor"),
        ("complement", "complement"),
        ("granzyme", "granzyme"),
        ("subtilisin", "subtilisin"),
        ("proteinase_k", "proteinase k"),
        ("enterokinase", "enterokinase"),
        ("u-plasminogen", "plasmin"),
        ("t-plasminogen", "plasminogen"),
        ("acrosin", "acrosin"),
        ("cathepsin", "cathepsin"),
    ]
    for label, kw in keywords:
        if kw in name_l:
            return label
    return "other_S8" if family == "S8_subtilisin" else "other_S1"


def uniprot_fetch(query: str) -> list[dict]:
    records = []
    params = {
        "query": query,
        "format": "json",
        "size": "500",
        "fields": "accession,id,protein_name,gene_primary,organism_name,organism_id,"
        "ec,length,ft_act_site,cc_subcellular_location,sequence,xref_pdb",
    }
    url = UNIPROT_SEARCH
    while url:
        r = SESSION.get(url, params=params if url == UNIPROT_SEARCH else None, timeout=60)
        r.raise_for_status()
        payload = r.json()
        records.extend(payload.get("results", []))
        url = r.links.get("next", {}).get("url")
        params = None
        time.sleep(0.2)
    return records


def parse_pdb_xrefs(entry: dict) -> list[dict]:
    out = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "PDB":
            continue
        props = {p["key"]: p["value"] for p in xref.get("properties", [])}
        res_s = props.get("Resolution", "").replace(" A", "").strip()
        try:
            resolution = float(res_s) if res_s else None
        except ValueError:
            resolution = None
        out.append(
            {
                "pdb_id": xref["id"].lower(),
                "method": props.get("Method", ""),
                "resolution": resolution,
                "chains": props.get("Chains", ""),
            }
        )
    return out


def pick_representative(xrefs: list[dict]) -> dict | None:
    def score(x):
        method = x["method"].lower()
        method_rank = 0 if "x-ray" in method else 1 if "em" in method else 2
        res = x["resolution"] if x["resolution"] is not None else 99.0
        return (method_rank, res)

    xray = [x for x in xrefs if x["resolution"] is not None and x["resolution"] <= 3.2]
    pool = xray or [x for x in xrefs if x["resolution"] is not None] or xrefs
    if not pool:
        return None
    return sorted(pool, key=score)[0]


def parse_chain_spec(chains: str) -> tuple[list[str], int, int] | None:
    s = chains.replace(" ", "")
    if not s:
        return None
    if "=" in s:
        ids_part, _, span_part = s.partition("=")
        m = re.search(r"(\d+)-(\d+)", span_part)
        start, end = (int(m.group(1)), int(m.group(2))) if m else (1, 9999)
    else:
        ids_part, start, end = s, 1, 9999
    ids = [c for c in re.split(r"[/,]", ids_part) if re.fullmatch(r"[A-Za-z0-9]+", c)]
    if not ids:
        return None
    return ids, start, end


def download_structure(pdb_id: str) -> Path | None:
    cif, pdb = STRUCT / f"{pdb_id}.cif", STRUCT / f"{pdb_id}.pdb"
    if cif.exists() or pdb.exists():
        return cif if cif.exists() else pdb
    for fmt, path in (("cif", cif), ("pdb", pdb)):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{fmt}"
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 200:
                path.write_bytes(r.content)
                return path
        except requests.RequestException:
            continue
    return None


def _resolve_chain(model, candidates: list[str]):
    chains = list(model)
    by_exact = {c.id: c for c in chains}
    for cid in candidates:
        if cid in by_exact:
            return by_exact[cid], cid, "exact"
    by_strip = {str(c.id).strip(): c for c in chains}
    for cid in candidates:
        if cid in by_strip:
            return by_strip[cid], cid, "stripped"
    by_lower = {str(c.id).strip().lower(): c for c in chains}
    for cid in candidates:
        if cid.lower() in by_lower:
            return by_lower[cid.lower()], cid, "case_insensitive"
    raise ChainNotFound(
        f"requested {candidates}, file has {[c.id for c in chains][:12]}"
    )


def extract_chain(path: Path, candidates: list[str]):
    parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, path)
    model = next(structure.get_models())
    chain, matched_id, how = _resolve_chain(model, candidates)

    seq, cas = [], []
    for res in chain:
        if res.id[0].strip() != "":
            continue
        name = res.get_resname().strip().upper()
        aa = AA3.get(name)
        if aa is None or "CA" not in res:
            continue
        seq.append(aa)
        cas.append((res.id[1], res["CA"].coord.astype(float).tolist()))

    nums = [c[0] for c in cas]
    diffs = [b - a for a, b in zip(nums, nums[1:]) if b > a]
    gaps = [d - 1 for d in diffs if d > 1]
    n_missing = sum(gaps)
    stats = {
        "n_missing": n_missing,
        "n_gaps": len(gaps),
        "max_gap": max(gaps) if gaps else 0,
        "coverage": len(cas) / (len(cas) + n_missing) if cas else 0.0,
        "chain_match": how,
        "chain_used": matched_id,
    }
    return "".join(seq), cas, stats


def _kingdom_by_name(organism: str) -> str:
    o = organism.lower()
    bacteria = ("bacteria", "bacillus", "streptomyces", "pseudomonas", "vibrio",
                "escherichia", "staphylococcus", "streptococcus", "lysobacter",
                "acheronobacter", "thermoactinomyces", "nonomuraea",
                "thermus", "geobacillus", "alkalihalobacillus", "paenibacillus",
                "xanthomonas", "serratia", "achromobacter")
    fungi = ("fungus", "aspergillus", "tritirachium", "saccharomyces", "fusarium",
             "trichoderma", "metarhizium", "engyodontium", "candida",
             "penicillium", "cordyceps")
    plants = ("arabidopsis", "cucumis", "oryza", "solanum", "glycine",
              "nicotiana", "zea mays", "hordeum", "triticum")
    if any(b in o for b in bacteria):
        return "Bacteria"
    if any(f in o for f in fungi):
        return "Fungi"
    if any(p in o for p in plants):
        return "Viridiplantae"
    if "virus" in o:
        return "Viruses"
    return "Eukaryota_other"


def kingdom_of(organism: str, lineage: list[str] | None = None) -> str:
    if lineage:
        L = set(lineage)
        if "Viruses" in L:
            return "Viruses"
        if "Archaea" in L:
            return "Archaea"
        if "Bacteria" in L:
            return "Bacteria"
        if "Eukaryota" in L:
            if "Fungi" in L:
                return "Fungi"
            if "Metazoa" in L:
                return "Metazoa"
            if "Viridiplantae" in L:
                return "Viridiplantae"
            return "Eukaryota_other"
    print(f"  ! no lineage for {organism!r}; falling back to name matching")
    return _kingdom_by_name(organism)


def superkingdom_of(organism: str, lineage: list[str] | None = None) -> str:
    if lineage:
        for d in ("Viruses", "Archaea", "Bacteria", "Eukaryota"):
            if d in set(lineage):
                return d
    k = _kingdom_by_name(organism)
    return {"Bacteria": "Bacteria", "Fungi": "Eukaryota", "Plant": "Eukaryota",
            "Virus": "Viruses"}.get(k, "Eukaryota")


def write_ca_pdb(seq: str, cas: list, out: Path) -> None:
    with out.open("w") as fh:
        for i, ((_, xyz), aa) in enumerate(zip(cas, seq), 1):
            fh.write(
                "ATOM  %5d  CA  %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                % (i, AA1TO3.get(aa, "GLY"), i, xyz[0], xyz[1], xyz[2])
            )
        fh.write("TER\nEND\n")


def tm_align_pair(a: Path, b: Path) -> float | None:
    try:
        r = subprocess.run([TMALIGN, str(a), str(b)],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    scores = [float(m) for m in re.findall(r"TM-score=\s*([\d.]+)", r.stdout)]
    return sum(scores[:2]) / len(scores[:2]) if scores else None


def fold_qc(meta: pd.DataFrame) -> pd.DataFrame:
    if not Path(TMALIGN).exists():
        print(f"\n  !! WARNING: TM-align not found at {TMALIGN}")
        print("  !! Fold QC SKIPPED -- cohort is NOT verified for fold membership.")
        print("  !! Set $TMALIGN or compile: g++ -O3 -ffast-math -o TMalign TMalign.cpp")
        meta["fold_tm"] = pd.NA
        meta["fold_qc"] = "not_run"
        return meta

    scores: dict[str, float] = {}
    for fold, grp in meta.groupby("fold"):
        med = grp["length"].median()
        panel = grp.reindex((grp["length"] - med).abs().sort_values().index)
        panel = panel.head(PANEL_SIZE)["id"].tolist()
        print(f"  fold={fold}: panel {panel}")
        for pid in grp["id"]:
            vals = [
                tm for ref in panel if ref != pid
                if (tm := tm_align_pair(CA_PDB / f"{pid}.pdb", CA_PDB / f"{ref}.pdb"))
                is not None
            ]
            scores[pid] = float(pd.Series(vals).median()) if vals else float("nan")

    meta["fold_tm"] = meta["id"].map(scores)
    meta["fold_qc"] = [
        "pass" if (pd.notna(t) and t >= MIN_FOLD_TM) else "fail" for t in meta["fold_tm"]
    ]
    return meta


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 1.0
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    return float((chi2 / (n * (min(tab.shape) - 1))) ** 0.5)


def main() -> None:
    rows, rejected = [], []

    def reject(acc, why, **extra):
        rejected.append({"uniprot": acc, "reason": why, **extra})

    for family, query in QUERIES.items():
        all_q = [query] + SUPPLEMENTARY_QUERIES.get(family, [])
        merged: dict[str, dict] = {}
        for qi, q in enumerate(all_q):
            tag = "primary" if qi == 0 else f"supplementary#{qi}"
            print(f"Fetching UniProt: {family} [{tag}]")
            recs = uniprot_fetch(q)
            print(f"  {len(recs)} reviewed entries with PDB")
            for rec in recs:
                merged.setdefault(rec["primaryAccession"], rec)
        records = list(merged.values())
        print(f"  -> {len(records)} unique accessions for {family}")
        (RAW / f"{family}_uniprot.json").write_text(json.dumps(records, indent=None))

        for rec in records:
            acc = rec["primaryAccession"]
            name = rec.get("proteinDescription", {}).get("recommendedName", {}).get(
                "fullName", {}
            ).get("value") or rec.get("uniProtkbId", acc)
            genes = rec.get("genes") or []
            gene = genes[0]["geneName"]["value"] if genes and "geneName" in genes[0] else ""
            org = rec.get("organism", {}).get("scientificName", "")
            taxid = rec.get("organism", {}).get("taxonId", "")
            lineage = rec.get("organism", {}).get("lineage") or []
            ecs = [
                e.get("value")
                for e in rec.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("ecNumbers", [])
            ]

            rep = pick_representative(parse_pdb_xrefs(rec))
            if not rep:
                reject(acc, "no_usable_structure")
                continue

            spec = parse_chain_spec(rep["chains"])
            if spec is None:
                reject(acc, "unparseable_chain_spec", chains=rep["chains"])
                continue
            cand_ids, start, end = spec

            path = download_structure(rep["pdb_id"])
            if path is None:
                reject(acc, "download_failed", pdb_id=rep["pdb_id"])
                continue

            try:
                seq, cas, st = extract_chain(path, cand_ids)
            except ChainNotFound as exc:
                reject(acc, "chain_not_found", pdb_id=rep["pdb_id"], detail=str(exc))
                print(f"  skip {acc} {rep['pdb_id']} {cand_ids}: chain absent")
                continue
            except Exception as exc:
                reject(acc, "parse_error", pdb_id=rep["pdb_id"], detail=str(exc)[:200])
                continue

            if not (MIN_OBS_LEN <= len(seq) <= MAX_OBS_LEN) or len(cas) < MIN_CA:
                reject(acc, "length_filter", length=len(seq), n_ca=len(cas))
                continue

            if st["max_gap"] > MAX_GAP or st["n_missing"] > MAX_MISSING \
                    or st["coverage"] < MIN_COVERAGE:
                reject(acc, "chain_breaks", pdb_id=rep["pdb_id"],
                       n_missing=st["n_missing"], max_gap=st["max_gap"],
                       coverage=round(st["coverage"], 3))
                print(f"  skip {acc} {rep['pdb_id']}: "
                      f"{st['n_missing']} missing, max gap {st['max_gap']}")
                continue

            rows.append(
                {
                    "id": f"{acc}_{rep['pdb_id']}_{st['chain_used']}",
                    "uniprot": acc,
                    "gene": gene,
                    "protein_name": name,
                    "organism": org,
                    "taxid": taxid,
                    "kingdom": kingdom_of(org, lineage),
                    "superkingdom": superkingdom_of(org, lineage),
                    "lineage": ";".join(lineage),
                    "family": family,
                    "fold": "trypsin" if family == "S1_trypsin" else "subtilisin",
                    "subclass": classify_subclass(ecs, name, family),
                    "ec": ";".join(ecs),
                    "pdb_id": rep["pdb_id"],
                    "chain_id": st["chain_used"],
                    "chain_match": st["chain_match"],
                    "method": rep["method"],
                    "resolution": rep["resolution"],
                    "uniprot_span": f"{start}-{end}",
                    "length": len(seq),
                    "cys_count": seq.count("C"),
                    "n_missing": st["n_missing"],
                    "n_gaps": st["n_gaps"],
                    "max_gap": st["max_gap"],
                    "coverage": round(st["coverage"], 4),
                    "sequence": seq,
                    "structure_path": str(path.relative_to(ROOT)),
                    "ca_json": json.dumps(cas),
                }
            )
            print(f"  kept {acc} {rep['pdb_id']}.{st['chain_used']} "
                  f"n={len(seq)} cov={st['coverage']:.2f} {name[:48]}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No proteins collected")
    df = df.drop_duplicates(subset=["uniprot"]).reset_index(drop=True)

    for keys, label in ((["pdb_id", "chain_id"], "duplicate_pdb_chain"),
                        (["sequence"], "duplicate_sequence")):
        dup = df[df.duplicated(subset=keys, keep="first")]
        for _, r in dup.iterrows():
            kept = df[~df.index.isin(dup.index)]
            twin = kept[kept[keys[0]] == r[keys[0]]]["id"]
            reject(r["uniprot"], label, pdb_id=r["pdb_id"],
                   detail=f"same as {twin.iloc[0] if len(twin) else '?'}")
            print(f"  drop {r['id']}: {label}")
        df = df.drop(index=dup.index).reset_index(drop=True)

    before = len(df)
    ranked = df.sort_values("resolution", na_position="last")
    keep_idx = ranked.groupby(["fold", "organism"], group_keys=False).head(ORG_CAP).index
    for _, r in df.loc[df.index.difference(keep_idx)].iterrows():
        reject(r["uniprot"], "organism_cap", pdb_id=r["pdb_id"],
               detail=f"{r['organism']} over ORG_CAP={ORG_CAP}")
    df = ranked.loc[keep_idx].reset_index(drop=True)
    if len(df) < before:
        print(f"\norganism cap (ORG_CAP={ORG_CAP}): {before} -> {len(df)} "
              f"({before - len(df)} logged to proteins_rejected.csv)")

    print(f"\nWriting CA-only PDBs -> {CA_PDB}")
    for _, row in df.iterrows():
        write_ca_pdb(row["sequence"], json.loads(row["ca_json"]),
                     CA_PDB / f"{row['id']}.pdb")

    print(f"\nFold QC via TM-align (threshold {MIN_FOLD_TM}, panel {PANEL_SIZE})")
    df = fold_qc(df)
    df[["id", "fold", "length", "fold_tm", "fold_qc"]].to_csv(
        DATA / "fold_qc.csv", index=False)
    failed = df[df["fold_qc"] == "fail"]
    for _, r in failed.iterrows():
        print(f"  FAIL {r['id']}  median TM to {r['fold']} panel = {r['fold_tm']:.3f}")
        reject(r["uniprot"], "fold_qc_fail", pdb_id=r["pdb_id"],
               fold_tm=round(float(r["fold_tm"]), 3))
    df = df[df["fold_qc"] != "fail"].reset_index(drop=True)

    coords = {row["id"]: json.loads(row["ca_json"]) for _, row in df.iterrows()}
    (DATA / "ca_coords.json").write_text(json.dumps(coords))

    meta = df.drop(columns=["ca_json"])
    meta.to_csv(DATA / "proteins.csv", index=False)
    pd.DataFrame(rejected).to_csv(DATA / "proteins_rejected.csv", index=False)

    with (DATA / "proteins.fasta").open("w") as fh:
        for _, row in meta.iterrows():
            fh.write(f">{row['id']}\n{row['sequence']}\n")

    print(f"\nWrote {len(meta)} proteins -> {DATA / 'proteins.csv'}")
    print(f"Rejected {len(rejected)} -> {DATA / 'proteins_rejected.csv'}")
    if rejected:
        print(pd.DataFrame(rejected)["reason"].value_counts().to_string())

    print("\n-- fold x subclass --")
    print(meta.groupby(["fold", "subclass"]).size().to_string())

    print("\n-- fold x superkingdom --")
    print(pd.crosstab(meta["fold"], meta["superkingdom"]).to_string())
    print("\n-- fold x kingdom --")
    print(pd.crosstab(meta["fold"], meta["kingdom"]).to_string())
    for lab in ("superkingdom", "kingdom"):
        v = cramers_v(meta["fold"], meta[lab])
        verdict = "WARNING" if v > 0.5 else "OK"
        print(f"\nCramer's V (fold ~ {lab}) = {v:.3f}   [{verdict}]")
    v = cramers_v(meta["fold"], meta["superkingdom"])
    if v > 0.5:
        print("  !! fold and superkingdom remain strongly associated.")
        print("  !! The different-fold control cannot be read as a fold effect.")
        print("  !! Recruit eukaryotic S8 (metazoan convertases, plant subtilases)")
        print("  !! and more bacterial S1 before trusting it.")
    else:
        print("  OK: fold and domain of life vary independently enough to separate.")

    print("\n-- chain quality --")
    print(meta[["length", "n_missing", "max_gap", "coverage"]].describe().to_string())
    if "fold_tm" in meta and meta["fold_tm"].notna().any():
        print("\n-- fold TM to own-fold panel --")
        print(meta.groupby("fold")["fold_tm"].describe()[
            ["count", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
