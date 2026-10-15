"""Plot accuracy vs. damage fraction from results/results_phase3.json.

    python analysis/plot_phase3.py results/results_phase3.json --out results/phase3_damage.png

x = fraction of one hemisphere silenced (0 = clean), y = mean of global and local accuracy.
Thin dots are individual seeds, thick line is the mean over seeds.
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = [
    ("aug=0 gate=fixed", "No damage training (fixed bridge)", "#888888"),
    ("aug=1 gate=fixed", "Damage training (fixed bridge)", "#1f77b4"),
    ("aug=1 gate=sev", "Damage training + severity gate", "#d95f02"),
]
X = [0, 20, 50, 80]


def seed_curve(r):
    return [(r["acc_g"] + r["acc_l"]) / 2, r["les20"], r["les50"], r["les80"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="results/phase3_damage.png")
    a = ap.parse_args()
    rs = json.load(open(a.path))
    labels = {r["label"] for r in rs}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (lab, name, col) in enumerate(CONDS):
        if lab not in labels:
            print(f"warning: '{lab}' not found, available: {sorted(labels)}")
            continue
        runs = np.array([seed_curve(r) for r in rs if r["label"] == lab])
        jitter = (i - 1) * 0.8
        for y in runs:
            ax.plot(np.array(X) + jitter, y, "o", color=col, alpha=0.3, ms=4)
        ax.plot(X, runs.mean(0), "-o", color=col, lw=2.5, ms=6, label=f"{name} (n={len(runs)})")

    ax.set_xlabel("Damage: % of one hemisphere's hidden units silenced at test time")
    ax.set_ylabel("Accuracy (mean of global and local)")
    ax.set_xticks(X)
    ax.set_xticklabels([f"{x}%" for x in X])
    ax.set_title("Training with random damage helps under heavy damage;\nthe severity-aware gate adds nothing distinguishable", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
