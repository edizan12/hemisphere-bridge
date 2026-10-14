#!/usr/bin/env python
"""
navon_hemi.py - Two identical "hemisphere" networks joined by a narrow bridge.

Question: if two identically-built networks exchange messages over a narrow bridge
for a few steps and are trained with one shared output head, does a division of
labour (global vs. local) emerge spontaneously? How do bridge width, delay, noise
and identical initialisation affect it? After damage, does a severity-aware gate
compensate better than a fixed bridge?

Usage (Colab or local):
    python navon_hemi.py --exp smoke                  # quick pipeline check
    python navon_hemi.py --exp phase1 --seeds 5       # bridge-width sweep
    python navon_hemi.py --exp symmetry --seeds 5     # what breaks the symmetry?
    python navon_hemi.py --exp symcheck --seeds 3 --deterministic
    python navon_hemi.py --exp phase3 --seeds 5       # lesion / compensation gate
Results are written to <out>/results_<exp>.json after every run.
"""
import argparse, json, os, sys, time, warnings
from dataclasses import dataclass, asdict, replace

if "--deterministic" in sys.argv:   # must be set BEFORE importing torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
GRID, CELL = 6, 8          # global shape = 6x6 cells, each cell holds an 8x8 local digit -> 48x48


# --------------------------------------------------------------------------
# DATA: Navon figures built from MNIST (a large digit made of small digits).
# The global and local labels are drawn independently; nothing tells the model
# which hemisphere should handle which label.
# --------------------------------------------------------------------------
def load_mnist(root):
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return (tr.data.float() / 255., tr.targets), (te.data.float() / 255., te.targets)


def make_navon(X, y, n, seed):
    g = torch.Generator().manual_seed(seed)
    ig = torch.randint(0, len(X), (n,), generator=g)
    il = torch.randint(0, len(X), (n,), generator=g)
    gm = F.adaptive_avg_pool2d(X[ig][:, None], GRID)[:, 0]               # n,6,6
    gm = (gm > 0.25 * gm.amax(dim=(1, 2), keepdim=True)).float()
    loc = F.adaptive_avg_pool2d(X[il][:, None], CELL)[:, 0]              # n,8,8
    loc = loc / loc.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    tile = loc.repeat(1, GRID, GRID)                                     # n,48,48
    mask = gm.repeat_interleave(CELL, 1).repeat_interleave(CELL, 2)      # n,48,48
    return (tile * mask)[:, None], y[ig], y[il]


def build_data(args):
    (Xa, ya), (Xb, yb) = load_mnist(args.data_root)
    out = {}
    for name, (X, y), n, s in (("train", (Xa, ya), args.ntrain, 0), ("test", (Xb, yb), args.ntest, 1)):
        img, gl, ll = make_navon(X, y, n, s)
        out[name] = (img.to(dev), gl.to(dev), ll.to(dev))
    return out


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
@dataclass
class Cfg:
    d: int = 64            # hidden size of each hemisphere
    k: int = 4             # bridge width; 0 = no bridge; d = "full" bridge
    T: int = 4             # number of message-passing steps
    dLR: int = 1           # L->R delay (steps, >= 1)
    dRL: int = 1           # R->L delay
    noise: float = 0.0     # independent noise added to the hidden state during training
    tied: bool = False     # True: both hemispheres start with exactly the same weights
    gate: str = "fixed"    # fixed | scalar | sev  (bridge gain gate)
    single: bool = False   # control: one large network (no bridge)
    lesion_aug: bool = False   # apply random damage during training
    epochs: int = 12
    bs: int = 256
    lr: float = 2e-3


class Enc(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.c = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(3))
        s.fc = nn.Linear(32 * 9, d)

    def forward(s, x):
        return s.fc(s.c(x).flatten(1))


class Bridge(nn.Module):
    """Linear d -> k -> d bottleneck: the message is confined to a k-dimensional subspace."""
    def __init__(s, d, k):
        super().__init__()
        s.down, s.up = nn.Linear(d, k), nn.Linear(k, d)

    def forward(s, h):
        return s.up(s.down(h))


class Gate(nn.Module):
    """Bridge gains (L->R, R->L). In 'sev' mode the damage severity is the input."""
    def __init__(s, mode):
        super().__init__()
        s.mode = mode
        if mode == "scalar":
            s.p = nn.Parameter(torch.zeros(2))
        elif mode == "sev":
            s.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
            nn.init.zeros_(s.net[2].weight)
            nn.init.zeros_(s.net[2].bias)

    def forward(s, sevL, sevR):
        B = sevL.shape[0]
        if s.mode == "fixed":
            o = torch.ones(B, 2, device=sevL.device)
        elif s.mode == "scalar":
            o = (2 * torch.sigmoid(s.p)).expand(B, 2)
        else:
            o = 2 * torch.sigmoid(s.net(torch.stack([sevL, sevR], 1)))
        return o[:, 0:1], o[:, 1:2]


def _vec(v, B, device):
    if v is None:
        return torch.zeros(B, device=device)
    if torch.is_tensor(v):
        return v.float().expand(B) if v.dim() == 0 else v.float()
    return torch.full((B,), float(v), device=device)


class Net(nn.Module):
    def __init__(s, c: Cfg):
        super().__init__()
        assert c.dLR >= 1 and c.dRL >= 1
        s.c, s.single = c, c.single
        s.d = 2 * c.d if c.single else c.d
        d = s.d
        s.encL, s.cellL = Enc(d), nn.GRUCell(2 * d, d)
        if not s.single:
            s.encR, s.cellR = Enc(d), nn.GRUCell(2 * d, d)
            s.brLR = Bridge(d, c.k) if c.k > 0 else None
            s.brRL = Bridge(d, c.k) if c.k > 0 else None
        s.gate = Gate(c.gate)
        s.head = nn.Sequential(nn.Linear(d if s.single else 2 * d, 64), nn.ReLU(), nn.Linear(64, 20))
        if c.tied and not s.single:
            s.encR.load_state_dict(s.encL.state_dict())
            s.cellR.load_state_dict(s.cellL.state_dict())
            if s.brLR is not None:
                s.brRL.load_state_dict(s.brLR.state_dict())
            with torch.no_grad():
                W = s.head[0].weight
                W[:, d:] = W[:, :d]

    def _msg(s, hist, t, delay, bridge, gain, B, device):
        i = t - delay
        if bridge is None or i < 0:
            return torch.zeros(B, s.d, device=device)
        return gain * bridge(hist[i])

    def _post(s, h, m):
        if s.training and s.c.noise > 0:
            h = h + s.c.noise * torch.randn_like(h)
        return h if m is None else h * m

    def forward(s, x, mL=None, mR=None, sevL=None, sevR=None, return_h=False):
        B, dv = x.shape[0], x.device
        sevL, sevR = _vec(sevL, B, dv), _vec(sevR, B, dv)
        zL = s.encL(x)
        hL = s._post(torch.tanh(zL), mL)
        if s.single:
            h = hL
            hist = [h]
            for t in range(1, s.c.T + 1):
                h = s._post(s.cellL(torch.cat([zL, torch.zeros_like(zL)], 1), h), mL)
            out = s.head(h)
            return (out, h, None) if return_h else out
        zR = s.encR(x)
        hR = s._post(torch.tanh(zR), mR)
        hist = [[hL], [hR]]
        gLR, gRL = s.gate(sevL, sevR)
        for t in range(1, s.c.T + 1):
            mR_in = s._msg(hist[0], t, s.c.dLR, s.brLR, gLR, B, dv)   # from L to R
            mL_in = s._msg(hist[1], t, s.c.dRL, s.brRL, gRL, B, dv)   # from R to L
            nL = s._post(s.cellL(torch.cat([zL, mL_in], 1), hist[0][-1]), mL)
            nR = s._post(s.cellR(torch.cat([zR, mR_in], 1), hist[1][-1]), mR)
            hist[0].append(nL)
            hist[1].append(nR)
        hL, hR = hist[0][-1], hist[1][-1]
        out = s.head(torch.cat([hL, hR], 1))
        return (out, hL, hR) if return_h else out


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# --------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------
def train_run(c: Cfg, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(c).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=c.lr)
    X, gl, ll = data["train"]
    n = len(X)
    for ep in range(c.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, c.bs):
            idx = perm[i:i + c.bs]
            B = len(idx)
            mL = mR = sevL = sevR = None
            if c.lesion_aug:
                sev = torch.rand(B, device=dev) * 0.9
                sev = sev * (torch.rand(B, device=dev) > 0.3).float()      # 30% undamaged
                side = (torch.rand(B, device=dev) < 0.5).float()          # 1 -> L is damaged
                sevL, sevR = sev * side, sev * (1 - side)
                mL = (torch.rand(B, net.d, device=dev) >= sevL[:, None]).float()
                mR = (torch.rand(B, net.d, device=dev) >= sevR[:, None]).float()
            out = net(X[idx], mL, mR, sevL, sevR)
            loss = F.cross_entropy(out[:, :10], gl[idx]) + F.cross_entropy(out[:, 10:], ll[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, loss.item()


# --------------------------------------------------------------------------
# MEASUREMENTS
# --------------------------------------------------------------------------
def make_mask(d, frac, seed=0):
    m = torch.ones(1, d)
    k = int(round(frac * d))
    if k > 0:
        m[0, torch.randperm(d, generator=torch.Generator().manual_seed(seed))[:k]] = 0
    return m.to(dev)


@torch.no_grad()
def collect(net, X, mL=None, mR=None, sevL=0., sevR=0., bs=1000):
    net.eval()
    lg, hl, hr = [], [], []
    for i in range(0, len(X), bs):
        o, a, b = net(X[i:i + bs], mL, mR, sevL, sevR, return_h=True)
        lg.append(o); hl.append(a); hr.append(b)
    return torch.cat(lg), torch.cat(hl), (torch.cat(hr) if hr[0] is not None else None)


def accs(lg, gl, ll):
    return ((lg[:, :10].argmax(1) == gl).float().mean().item(),
            (lg[:, 10:].argmax(1) == ll).float().mean().item())


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    xy = (X.T @ Y).norm() ** 2
    return (xy / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-12)).item()


def probe(Ftr, ytr, Fte, yte, steps=300):
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    lin = nn.Linear(Ftr.shape[1], 10).to(Ftr.device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(Ftr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (lin(Fte).argmax(1) == yte).float().mean().item()


FRACS = (0.2, 0.5, 0.8)


def analyze(net, c: Cfg, data):
    Xtr, gtr, ltr = data["train"]
    Xte, gte, lte = data["test"]
    r = {}
    lg, hL, hR = collect(net, Xte)
    r["acc_g"], r["acc_l"] = accs(lg, gte, lte)
    ntp = min(10000, len(Xtr))
    _, tL, tR = collect(net, Xtr[:ntp])
    r["pLg"] = probe(tL, gtr[:ntp], hL, gte)
    r["pLl"] = probe(tL, ltr[:ntp], hL, lte)
    nan = float("nan")
    if hR is None:
        r.update(pRg=nan, pRl=nan, SI=nan, SIs=nan, cka=nan)
    else:
        r["pRg"] = probe(tR, gtr[:ntp], hR, gte)
        r["pRl"] = probe(tR, ltr[:ntp], hR, lte)
        s = ((r["pLg"] - r["pLl"]) - (r["pRg"] - r["pRl"])) / 2 / 0.9
        r["SIs"], r["SI"] = s, abs(s)          # SIs > 0: L is more "global", R more "local"
        r["cka"] = linear_cka(hL[:2000], hR[:2000])
    # lesion: silence a fraction of one hemisphere's hidden units
    les = {}
    for f in FRACS:
        for side in ("L", "R"):
            if c.single and side == "R":
                les[(side, f)] = (nan, nan)
                continue
            m = make_mask(net.d, f)
            kw = dict(mL=m, sevL=f) if side == "L" else dict(mR=m, sevR=f)
            les[(side, f)] = accs(collect(net, Xte, **kw)[0], gte, lte)
        r[f"les{int(f * 100)}"] = float(np.nanmean([np.mean(les[(s_, f)]) for s_ in ("L", "R")]))
    f = 0.5
    dgL, dlL = r["acc_g"] - les[("L", f)][0], r["acc_l"] - les[("L", f)][1]
    dgR, dlR = r["acc_g"] - les[("R", f)][0], r["acc_l"] - les[("R", f)][1]
    r["LIs"] = ((dgL - dlL) - (dgR - dlR)) / 2      # nan if single
    r["LI"] = abs(r["LIs"])
    r["gains"] = None
    if c.gate == "sev" and not c.single:
        with torch.no_grad():
            r["gains"] = {str(sv): [round(v.item(), 3) for v in net.gate(
                torch.tensor([sv], device=dev), torch.tensor([0.], device=dev))] for sv in (0., .2, .5, .8)}
    return r


# --------------------------------------------------------------------------
# EXPERIMENTS
# --------------------------------------------------------------------------
def exp_configs(name, base: Cfg):
    if name == "smoke":
        return [("single", replace(base, single=True)), ("no_bridge", replace(base, k=0)),
                ("k=4", base), ("k=4+sevgate+aug", replace(base, gate="sev", lesion_aug=True))]
    if name == "phase1":
        cs = [("single", replace(base, single=True)), ("no_bridge", replace(base, k=0))]
        cs += [(f"k={k}", replace(base, k=k)) for k in (2, 4, 8, 16)]
        cs += [(f"full(k={base.d})", replace(base, k=base.d))]
        return cs
    if name == "symmetry":
        cs = []
        for tied in (False, True):
            for noise in (0.0, 0.1):
                for dl in ((1, 1), (1, 2)):
                    cs.append((f"tied={int(tied)} noise={noise} delay={dl[0]}/{dl[1]}",
                               replace(base, tied=tied, noise=noise, dLR=dl[0], dRL=dl[1])))
        return cs
    if name == "symcheck":      # fully symmetric start: expect SI=0, CKA=1
        return [("tied noise=0 delay=1/1", replace(base, tied=True, noise=0.0, dLR=1, dRL=1))]
    if name == "phase3":
        return [(f"aug={int(a)} gate={g}", replace(base, lesion_aug=a, gate=g))
                for a in (False, True) for g in ("fixed", "scalar", "sev")]
    raise ValueError(name)


COLS = {
    "default": ["acc_g", "acc_l", "SI", "cka", "LI", "les50"],
    "phase3": ["acc_g", "acc_l", "les20", "les50", "les80"],
}


def summarize(results, cols):
    warnings.simplefilter("ignore")
    labels = list(dict.fromkeys(r["label"] for r in results))
    w = max(len(l) for l in labels) + 2
    print("\n" + "".ljust(w) + "".join(c.rjust(14) for c in cols) + "   n  params")
    for lab in labels:
        rs = [r for r in results if r["label"] == lab]
        row = ""
        for c in cols:
            v = np.array([r[c] for r in rs], dtype=float)
            row += f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f}".rjust(14)
        print(lab.ljust(w) + row + f"  {len(rs):2d}  {rs[0]['params']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="smoke", choices=["smoke", "phase1", "symmetry", "symcheck", "phase3"])
    ap.add_argument("--deterministic", action="store_true",
                    help="disable GPU numerical noise (slower); use for symcheck")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ntrain", type=int, default=20000)
    ap.add_argument("--ntest", type=int, default=5000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=4, help="bridge width for symmetry/phase3")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    if a.exp == "smoke":
        a.ntrain, a.ntest, a.seeds = 4000, 1000, 1
        a.epochs = a.epochs or 3
    if a.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(a.out, exist_ok=True)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    data = build_data(a)
    base = Cfg(d=a.d, k=a.k, T=a.T, epochs=a.epochs or Cfg.epochs)
    results, path = [], os.path.join(a.out, f"results_{a.exp}.json")
    for label, c in exp_configs(a.exp, base):
        for seed in range(a.seeds):
            t0 = time.time()
            net, last = train_run(c, data, seed)
            r = analyze(net, c, data)
            r.update(label=label, seed=seed, cfg=asdict(c), params=n_params(net), loss=last,
                     sec=round(time.time() - t0, 1))
            results.append(r)
            print(f"[{label} s{seed}] g={r['acc_g']:.3f} l={r['acc_l']:.3f} SI={r['SI']:.3f} "
                  f"cka={r['cka']:.3f} les50={r['les50']:.3f} ({r['sec']}s)", flush=True)
            if r["gains"]:
                print("   gate gains (damage severity -> [L->R, R->L], L damaged):", r["gains"])
            with open(path, "w") as fh:
                json.dump(results, fh, indent=1)
            del net
            if dev == "cuda":
                torch.cuda.empty_cache()
    summarize(results, COLS["phase3"] if a.exp == "phase3" else COLS["default"])
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
