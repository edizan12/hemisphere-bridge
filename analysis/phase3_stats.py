import json, sys
import numpy as np

rs = json.load(open(sys.argv[1]))
KEYS = ("acc_g", "acc_l", "les20", "les50", "les80")


def get(label, key):
    xs = sorted((r for r in rs if r["label"] == label), key=lambda r: r["seed"])
    return np.array([r[key] for r in xs], float)


def paired(a_label, b_label, key):
    d = get(a_label, key) - get(b_label, key)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12)
    return d.mean(), t, int((d > 0).sum()), n


labels = list(dict.fromkeys(r["label"] for r in rs))
print("TABLE (mean over seeds)")
print("".ljust(20) + "".join(k.rjust(8) for k in KEYS) + "   n")
for l in labels:
    print(l.ljust(20) + "".join(f"{get(l, k).mean():8.3f}" for k in KEYS) + f"  {len(get(l, 'acc_g'))}")

print("\nPAIRED DIFFERENCES (mean diff, t, seeds ahead / n); |t| > ~2.8 needed at 4 d.f.")
pairs = [("aug=1 gate=sev", "aug=1 gate=fixed"),
         ("aug=1 gate=scalar", "aug=1 gate=fixed"),
         ("aug=1 gate=fixed", "aug=0 gate=fixed")]
for a, b in pairs:
    print(f"{a}  minus  {b}")
    for k in KEYS:
        m, t, w, n = paired(a, b, k)
        print(f"   {k:6s} diff={m:+.3f}  t={t:+.2f}  ahead {w}/{n}")

print("\nGATE GAINS, aug=1 gate=sev, left damaged: [L->R, R->L] per seed")
for r in sorted((r for r in rs if r["label"] == "aug=1 gate=sev"), key=lambda r: r["seed"]):
    g = r.get("gains")
    if g:
        print(f"  seed {r['seed']}: " + "  ".join(f"sev {k}: {v}" for k, v in g.items()))
