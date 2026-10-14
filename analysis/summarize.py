"""Summarize a results_*.json file: per-condition mean +- std over seeds,
and (optionally) permutation tests against a reference condition.

    python analysis/summarize.py results/results_phase1.json --ref no_bridge
"""
import argparse, json
import numpy as np

COLS = ["acc_g", "acc_l", "SI", "cka", "LI", "les50"]


def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool, hits = np.r_[a, b], 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(pool[: len(a)].mean() - pool[len(a):].mean()) >= obs
    return (hits + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default=None, help="label to compare every other condition against (on SI)")
    a = ap.parse_args()
    rs = json.load(open(a.path))
    labels = list(dict.fromkeys(r["label"] for r in rs))
    by = {l: [r for r in rs if r["label"] == l] for l in labels}
    w = max(map(len, labels)) + 2
    print("".ljust(w) + "".join(c.rjust(14) for c in COLS) + "   n  SIs(+/-)")
    for l in labels:
        row = ""
        for c in COLS:
            v = np.array([r[c] for r in by[l]], float)
            row += ("nan" if np.isnan(v).all() else f"{np.nanmean(v):.3f}±{np.nanstd(v, ddof=1):.3f}").rjust(14)
        s = np.array([r["SIs"] for r in by[l]], float)
        sg = "" if np.isnan(s).all() else f"{int((s > 0).sum())}/{int((s < 0).sum())}"
        print(l.ljust(w) + row + f"  {len(by[l]):2d}  {sg}")
    if a.ref:
        ref = [r["SI"] for r in by[a.ref]]
        print(f"\nSI permutation tests vs '{a.ref}' (two-sided, 20k permutations)")
        for l in labels:
            if l in (a.ref, "single"):
                continue
            x = [r["SI"] for r in by[l]]
            print(f"  {l.ljust(w)} diff={np.mean(ref) - np.mean(x):+.3f}  p={perm_p(ref, x):.3f}")


if __name__ == "__main__":
    main()
