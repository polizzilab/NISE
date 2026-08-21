#!/usr/bin/env python3
"""Rerank a finished NISE run by buried, non-H-bonded polar atoms (BUNs).

NISE expands each round on ligand confidence (pLDDT). To select final leads, the paper reranks the
self-consistent designs by a linear combination of buried non-H-bonded polar atoms of the protein and
ligand (the reranking behind the EPIC and PiB selections; Fry et al., Extended Data Fig. 3). This scores
each predicted complex with bunsalyze and sorts by fewest BUNs.

bunsalyze (https://github.com/polizzilab/bunsalyze) is an optional companion; install it to use this.
"""
import argparse
import csv
import sys
from pathlib import Path


def score(pdb, smiles, sasa_threshold, alpha_hull_alpha):
    from bunsalyze.bunsalyze import main as run
    r = run(str(pdb), smiles, sasa_threshold=sasa_threshold, alpha_hull_alpha=alpha_hull_alpha)
    return r["buns_score"], len(r["ligand_buns"]), len(r["protein_buns"])


def main():
    ap = argparse.ArgumentParser(description="Rerank NISE designs by buried non-H-bonded polar atoms (BUNs), per Fry et al.")
    ap.add_argument("input_dir", help="a completed NISE output directory")
    ap.add_argument("smiles", help="ligand SMILES (the one passed to NISE)")
    ap.add_argument("--output", default=None, help="reranked CSV (default: <input_dir>/reranked_by_buns.csv)")
    ap.add_argument("--pdb-glob", default="**/predictions/*/*_model_0.pdb",
                    help="glob under input_dir for the self-consistent predicted complexes")
    ap.add_argument("--sasa-threshold", type=float, default=2.5)
    ap.add_argument("--alpha-hull-alpha", type=float, default=14.0)
    args = ap.parse_args()

    try:
        import bunsalyze  # noqa: F401
    except ImportError:
        sys.exit("rerank_by_buns needs bunsalyze: pip install git+https://github.com/polizzilab/bunsalyze")

    paths = sorted(str(p) for p in Path(args.input_dir).glob(args.pdb_glob))
    if not paths:
        sys.exit(f"no complexes matching '{args.pdb_glob}' under {args.input_dir}")

    rows = []
    for pdb in paths:
        try:
            s, lig, prot = score(pdb, args.smiles, args.sasa_threshold, args.alpha_hull_alpha)
            rows.append({"design": pdb, "buns_score": s, "ligand_buns": lig, "protein_buns": prot, "buns_error": ""})
        except Exception as e:                          # surface failures, never silently drop a complex
            rows.append({"design": pdb, "buns_score": float("inf"), "ligand_buns": "", "protein_buns": "", "buns_error": str(e)})
    rows.sort(key=lambda r: r["buns_score"])            # fewest first; buns_score = 2*ligand_buns + protein_buns

    out = args.output or str(Path(args.input_dir) / "reranked_by_buns.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["design", "buns_score", "ligand_buns", "protein_buns", "buns_error"])
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["buns_error"]]
    print(f"reranked {len(rows)} complexes by BUNs ({len(ok)} scored) -> {out}")
    for r in ok[:10]:
        print(f"  buns {r['buns_score']:>4}  (lig {r['ligand_buns']}, prot {r['protein_buns']})  {r['design']}")
    if len(rows) - len(ok):
        print(f"[warn] {len(rows) - len(ok)} complex(es) failed bunsalyze; kept with buns_score=inf and a buns_error column")


if __name__ == "__main__":
    main()
