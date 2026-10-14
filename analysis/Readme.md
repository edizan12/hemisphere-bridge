# Two identical networks, one narrow bridge

A small, reproducible test of a hemisphere-inspired idea: if two identical networks share only a narrow
communication channel, do they spontaneously divide a task between them (one on local detail, one on global
shape) without being told which is which?

**Status: early, small-scale exploration. Not a finished study, and not a claim of novelty** (see
[Related work](#related-work)). The lesion/compensation experiment is implemented but **not yet run**.

## Setup

- **Task (Navon figures from MNIST).** 48×48 images: a large digit (global shape, a 6×6 binary mask) built
  from small 8×8 digit tiles (local). The global and local labels are drawn independently. The model predicts
  **both** labels with one shared output head. No hemisphere is told which label to handle.
- **Model.** Two identical hemispheres, each a small CNN encoder (16-32-32 channels, hidden size d=64) plus a GRU
  cell. They run for T=4 message-passing steps. Each hemisphere receives the other's hidden state through a
  linear bottleneck (d→k→d), with a configurable per-direction delay (≥1 step). A shared MLP head reads both
  final hidden states.
- **Controls.** No bridge (k=0), full-width bridge (k=64), a single larger network (not parameter-matched, see
  caveats), identical vs. independent initialisation, added hidden-state noise, asymmetric delay.
- **Measures.**
  - *Specialization index* (SI): from linear probes on each hemisphere's hidden state,
    `SIs = ((pLg − pLl) − (pRg − pRl)) / 2 / 0.9`, where `pXy` is the probe accuracy for label y (g = global,
    l = local) from hemisphere X (L/R). We report |SI| (0 = same, ~1 = fully split); the sign says which side
    became "global".
  - *CKA*: linear CKA between the two hemispheres' hidden states (1 = identical representations).
  - *Lesion index* (LI): test-time silencing of 50% of one hemisphere's hidden units; compares how much each
    label's accuracy drops when the left vs. right hemisphere is damaged.
- 5 seeds per condition, 12 epochs, 20k training / 5k test images. Permutation tests are two-sided.

## Results (5 seeds per condition)

All numbers are mean ± std over seeds; full tables via `analysis/summarize.py`.

1. **The task is learned:** global accuracy ≈ 0.79, local ≈ 0.88 for two-hemisphere models.
2. **Specialization appears without assigned roles:** |SI| of roughly 0.2–0.35 across conditions, ranging from
   ~0 to ~0.5 across individual seeds.
3. **The narrow bridge does not drive it.** The no-bridge control specializes just as much
   (SI 0.24 ± 0.27) as k = 2/4/8/16 (0.33 / 0.25 / 0.23 / 0.21; permutation p = 0.53–0.99 vs. no bridge). A
   full-width bridge trends lower (SI 0.13 ± 0.11) but this is not significant at 5 seeds.
4. **Which side takes which role is random:** it varies with the seed (initialisation), not with the bridge
   width; signs split roughly evenly.
5. **Symmetry breaking.** With identical initial weights and deterministic GPU kernels, the control (no noise,
   equal delays) stays exactly symmetric (SI = 0.001, CKA = 1.000, 5 seeds). Starting from that symmetric state:
   - an asymmetric delay alone (one direction takes 2 steps) gives SI 0.154 ± 0.092 (all 5 seeds above control;
     exact permutation p = 0.008);
   - noise alone gives SI 0.234 ± 0.114 (p ≈ 0.006); both together 0.247 ± 0.067;
   - the delay does **not** determine which side becomes global: pooled over all delayed conditions the sign
     split is 10 vs. 10.
   Independent random initialisation already specializes (SI ≈ 0.27–0.34), and adding delay or noise on top does
   not increase it noticeably.
6. Two hemispheres beat a single network in accuracy (local 0.883 vs. 0.805, global 0.790 vs. 0.777), but this
   comparison is confounded by capacity (see caveats) and is not specific to the bridge.

## Caveats

- One task, one architecture, one scale, 5 seeds. Effects of a few hundredths in SI are not resolvable; SI is an
  absolute value and is biased upward at small sample sizes (it has no zero floor under noise).
- The shared output head reads both hemispheres, which is a second channel between them. The narrow bridge is
  therefore not the only way information is combined, and this may be why specialization does not depend on it.
- The single-network baseline is not capacity-matched: it has half the convolutional width of the pair but more
  recurrent parameters.
- **GPU non-determinism incident.** My first symmetry run used default CUDA kernels. In the "identical start, no
  noise" condition the supposedly symmetric networks diverged (SI 0.057, CKA 0.33) because of tiny floating-point
  differences amplified by the unstable symmetric state. After enabling deterministic mode (`--deterministic`)
  the control behaves exactly (SI ≈ 0, CKA = 1). The original run is kept as
  `results/results_symmetry_gpu_nondeterministic.json` for transparency; use the deterministic file. The phase 1
  run used independent initialisations and default kernels.
- An earlier single-seed run suggested the bridge *reduces* specialization; with 5 seeds that was seed noise.

## Not done yet

- **Lesion-aware gate** (`--exp phase3`): train with random hemisphere damage and let the bridge gain depend on
  damage severity, to test the "compete when damage is mild, cooperate when severe" idea from the stroke
  literature. Code is included but untested at scale; no results yet.
- Longer training, more seeds, other tasks, matched baselines, and a proper literature review.

## Reproduce

```bash
pip install -r requirements.txt
python navon_hemi.py --exp smoke                                # quick pipeline check (small data, 3 epochs)
python navon_hemi.py --exp phase1 --seeds 5                     # bridge width sweep
python navon_hemi.py --exp symmetry --seeds 5 --k 4 --deterministic
python navon_hemi.py --exp symcheck --seeds 3 --deterministic   # symmetric-start sanity check
python analysis/summarize.py results/results_phase1.json --ref no_bridge
python analysis/summarize.py results/results_symmetry_deterministic.json
```

Tested on a GTX 1650 (4 GB), PyTorch 2.14 + CUDA 12.6; a run takes ~15 s (single) to ~35 s (two hemispheres).
MNIST is downloaded automatically to `./data`.

## Related work

Not an exhaustive review. Closely related earlier work I am aware of:

- Shevtsova & Reggia, "Lateralization in a bihemispheric neural model," *Neurocomputing* 26 (1999): emergent
  lateralization in a two-hemisphere letter-identification model, with built-in asymmetries and callosal
  inhibition.
- Shevtsova & Reggia, "Interhemispheric effects of simulated lesions in a bihemispheric neural model,"
  *Brain and Cognition* 44 (2000), and Reggia, Shkuro & Shevtsova (2001, LNCS 2036): simulated lesions; the
  intact hemisphere's contribution depends on pre-existing lateralization and lesion size.
- Rajagopalan, Rawlinson, Goldberg & Kowadlo, "Deep learning in a bilateral brain with hemispheric
  specialization," arXiv:2209.06862: two CNN hemispheres with local/global specialization induced by different
  training objectives.
- Jacobs & Kosslyn (1994): receptive-field size as a basis for complementary hemispheric representations.

This repository differs in setting (modern deep networks, no assigned roles, narrow bridge and delay as the
manipulated variables) but I have not verified that this exact combination is untested. Corrections welcome.
