# Two identical networks, one narrow bridge

A small, reproducible test of a hemisphere-inspired idea: if two identical networks share only a narrow communication channel, do they spontaneously divide a task between them (one on local detail, one on global shape) without being told which is which? And after damage, does a damage-aware bridge help?

**Status:** early, small-scale exploration. Not a finished study, and not a claim of novelty (see [Related work](#related-work)). Main findings, all at 5 seeds per condition:

1. The narrow bridge does **not** drive specialization; the no-bridge control specializes just as much.
2. Specialization is seed-dependent. Asymmetric delay or noise can break an exactly symmetric start, but neither decides which side becomes "global".
3. Training with random damage clearly improves accuracy under heavy damage (80% of one hemisphere silenced: ~0.62 to ~0.77).
4. A damage-severity-aware gate is **not** distinguishably better than a fixed bridge, and what it learns is the opposite of the "cooperate when damage is severe" hypothesis.

<img width="794" height="511" alt="image" src="https://github.com/user-attachments/assets/c65f50a9-816f-4e39-a13b-9b58a1b89c49" />


*Accuracy (mean of global and local) as a function of the fraction of one hemisphere silenced at test time. Thin dots: individual seeds; thick lines: mean over 5 seeds.*

## Setup

**Task (Navon figures from MNIST).** 48x48 images: a large digit (global shape, a 6x6 binary mask) built from small 8x8 digit tiles (local). The global and local labels are drawn independently. The model predicts both labels with one shared output head. No hemisphere is told which label to handle.

**Model.** Two identical hemispheres, each a small CNN encoder (16-32-32 channels, hidden size d=64) plus a GRU cell. They run for T=4 message-passing steps. Each hemisphere receives the other's hidden state through a linear bottleneck (d -> k -> d), with a configurable per-direction delay (>= 1 step). A shared MLP head reads both final hidden states.

**Bridge gate (phase 3).** The message gain of each direction can be `fixed` (1.0), `scalar` (one learned gain per direction) or `sev` (a small network that maps the damage severity of each hemisphere to the two gains).

**Controls.** No bridge (k=0), full-width bridge (k=64), a single larger network (not parameter-matched, see caveats), identical vs. independent initialisation, added hidden-state noise, asymmetric delay.

**Measures.**

- **Specialization index (SI):** from linear probes on each hemisphere's hidden state, `SIs = ((pLg - pLl) - (pRg - pRl)) / 2 / 0.9`, where `pXy` is the probe accuracy for label y (g = global, l = local) from hemisphere X (L/R). We report |SI| (0 = same, ~1 = fully split); the sign says which side became "global".
- **CKA:** linear CKA between the two hemispheres' hidden states (1 = identical representations).
- **Lesion index (LI):** test-time silencing of 50% of one hemisphere's hidden units; compares how much each label's accuracy drops when the left vs. right hemisphere is damaged.
- **Lesion accuracy (les20 / les50 / les80):** mean of global and local accuracy after silencing 20 / 50 / 80% of one hemisphere's hidden units, averaged over the two sides.

5 seeds per condition, 12 epochs, 20k training / 5k test images. Permutation tests are two-sided; paired comparisons use a paired t statistic over seeds.

## Results (5 seeds per condition)

All numbers are mean +- std over seeds; full tables via `analysis/summarize.py`.

### Phase 1: bridge width and specialization

- The task is learned: global accuracy ~0.79, local ~0.88 for two-hemisphere models.
- Specialization appears without assigned roles: |SI| of roughly 0.2-0.35 across conditions, ranging from ~0 to ~0.5 across individual seeds.
- **The narrow bridge does not drive it.** The no-bridge control specializes just as much (SI 0.24 +- 0.27) as k = 2/4/8/16 (0.33 / 0.25 / 0.23 / 0.21; permutation p = 0.53-0.99 vs. no bridge). A full-width bridge trends lower (SI 0.13 +- 0.11) but this is not significant at 5 seeds.
- Which side takes which role is random: it varies with the seed (initialisation), not with the bridge width; signs split roughly evenly.

### Symmetry breaking

With identical initial weights and deterministic GPU kernels, the control (no noise, equal delays) stays exactly symmetric (SI = 0.001, CKA = 1.000, 5 seeds). Starting from that symmetric state:

- an asymmetric delay alone (one direction takes 2 steps) gives SI 0.154 +- 0.092 (all 5 seeds above control; exact permutation p = 0.008);
- noise alone gives SI 0.234 +- 0.114 (p ~ 0.006); both together 0.247 +- 0.067;
- the delay does not determine which side becomes global: pooled over all delayed conditions the sign split is 10 vs. 10. Independent random initialisation already specializes (SI ~ 0.27-0.34), and adding delay or noise on top does not increase it noticeably.

Two hemispheres beat a single network in accuracy (local 0.883 vs. 0.805, global 0.790 vs. 0.777), but this comparison is confounded by capacity (see caveats) and is not specific to the bridge.

### Phase 3: lesion training and a severity-aware gate

Mean accuracy after silencing a fraction of one hemisphere's hidden units at test time (`aug=1` means random damage was applied during training; k=4 bridge):

| condition    | acc_g | acc_l | les20 | les50 | les80 |
|--------------|-------|-------|-------|-------|-------|
| aug=0 fixed  | 0.786 | 0.868 | 0.817 | 0.770 | 0.624 |
| aug=0 scalar | 0.791 | 0.885 | 0.825 | 0.780 | 0.628 |
| aug=0 sev    | 0.789 | 0.863 | 0.812 | 0.769 | 0.633 |
| aug=1 fixed  | 0.787 | 0.865 | 0.824 | 0.806 | 0.769 |
| aug=1 scalar | 0.788 | 0.860 | 0.818 | 0.804 | 0.764 |
| aug=1 sev    | 0.786 | 0.881 | 0.829 | 0.818 | 0.773 |

Paired comparisons over the 5 seeds (about |t| = 2.8 is needed for p < 0.05 with 4 degrees of freedom):

| comparison | acc_l | les20 | les50 | les80 |
|------------|-------|-------|-------|-------|
| `aug=1 fixed` minus `aug=0 fixed` | -0.002 (t = -0.1) | +0.007 (t = 0.7) | +0.036 (t = 3.9) | +0.146 (t = 13.1) |
| `aug=1 sev` minus `aug=1 fixed` | +0.016 (t = 0.8) | +0.005 (t = 0.6) | +0.012 (t = 1.3) | +0.003 (t = 0.2) |
| `aug=1 scalar` minus `aug=1 fixed` | -0.006 (t = -0.5) | -0.006 (t = -0.8) | -0.003 (t = -0.3) | -0.005 (t = -0.6) |

- **Training with damage helps a lot under heavy damage.** There is no difference at 20% damage, a clear gain at 50% (+0.036, ahead in 5 of 5 seeds) and a large gain at 80% (+0.146, 5 of 5 seeds; all `aug=0` runs end below 0.67, all `aug=1` runs above 0.73). Clean accuracy is not measurably affected.
- **The severity-aware gate adds nothing distinguishable.** `aug=1 sev` is ahead of `aug=1 fixed` in 3 to 4 of 5 seeds depending on the lesion level, by at most ~0.012, with paired t-values of 0.2-1.3. The `scalar` gate behaves like `fixed`. The higher clean local accuracy of `aug=1 sev` (0.881 vs. 0.865) is within seed-to-seed spread and is not treated as a finding.
- **The gate learns the opposite of the hypothesis.** The idea was "compete when damage is mild, cooperate when severe", i.e. the intact side should open the channel toward the damaged side as damage grows. With the left hemisphere damaged, the gain from damaged to intact (L->R) rose with severity in 5 of 5 seeds (for example 0.73 to 0.97), while the gain from intact to damaged (R->L) fell in 4 of 5 seeds (for example 0.61 to 0.23); in the fifth it rose slightly (0.85 to 0.88). So the learned strategy is closer to "move the damaged side's information to the intact side, and throttle the flow back". Why this happens is not understood. The gate sees which side is damaged but not which hemisphere is the "global" one, which makes per-seed interpretation hard.
- Gate gains for `aug=0` rows are meaningless: the severity input is always 0 there, so any variation comes from initialisation.
- Phase 3 used default (non-deterministic) GPU kernels, so reruns give slightly different numbers. An earlier run of the same experiment gave the same qualitative picture (large benefit of damage training, gate within noise) but different third-digit values; only the numbers above match `results/results_phase3.json`.

## Caveats

- One task, one architecture, one scale, 5 seeds. Effects of a few hundredths in SI or accuracy are not resolvable; SI is an absolute value and is biased upward at small sample sizes (it has no zero floor under noise).
- The shared output head reads both hemispheres, which is a second channel between them. The narrow bridge is therefore not the only way information is combined, and this may be why specialization does not depend on it.
- The single-network baseline is not capacity-matched: it has half the convolutional width of the pair but more recurrent parameters.
- **Phase 3 has no single-network control trained with the same damage.** The robustness gain from `aug=1` is most likely generic damage-robust training (similar to dropout), not specific to the two-hemisphere bridge or to the gate. This is untested.
- The test-time lesion mask is one fixed random subset of units (seed 0) per damage level and is applied to one hemisphere at a time. Training damage is resampled for every example.
- **GPU non-determinism incident.** My first symmetry run used default CUDA kernels. In the "identical start, no noise" condition the supposedly symmetric networks diverged (SI 0.057, CKA 0.33) because of tiny floating-point differences amplified by the unstable symmetric state. After enabling deterministic mode (`--deterministic`) the control behaves exactly (SI ~ 0, CKA = 1). `results/results_symmetry.json` is the deterministic rerun. The phase 1 and phase 3 runs used independent initialisations and default kernels.
- An earlier single-seed run suggested the bridge reduces specialization; with 5 seeds that was seed noise.

## Not done yet

- A single network trained with the same random damage (needed to say whether the phase 3 gain is bridge-specific).
- More seeds for the gate comparison (`aug=1 fixed` vs. `aug=1 sev`); 5 seeds cannot resolve a difference of ~0.01.
- Longer training, other tasks, parameter-matched baselines, a proper literature review.

## Reproduce

```bash
pip install -r requirements.txt
python analysis/navon_hemi.py --exp smoke --out results                       # quick pipeline check (small data, 3 epochs)
python analysis/navon_hemi.py --exp phase1 --seeds 5 --out results            # bridge width sweep
python analysis/navon_hemi.py --exp symmetry --seeds 5 --k 4 --deterministic --out results
python analysis/navon_hemi.py --exp symcheck --seeds 3 --deterministic --out results   # symmetric-start sanity check
python analysis/navon_hemi.py --exp phase3 --seeds 5 --out results            # lesion training / gate comparison

python analysis/summarize.py results/results_phase1.json --ref no_bridge
python analysis/summarize.py results/results_symmetry.json
python analysis/phase3_stats.py results/results_phase3.json                   # phase 3 table, paired t-values, gate gains
python analysis/plot_phase3.py results/results_phase3.json --out results/phase3_damage.png
```

The phase 3 table is also printed at the end of the `phase3` run.

Tested on a GTX 1650 (4 GB), PyTorch 2.14 + CUDA 12.6; a run takes ~15 s (single) to ~35 s (two hemispheres). The script uses the GPU automatically when `torch.cuda.is_available()` is true (first line of output: `device: cuda ...`). MNIST is downloaded automatically to `./data`.

## Results files

| file | experiment |
|------|------------|
| `results/results_phase1.json` | bridge-width sweep (independent initialisation) |
| `results/results_symmetry.json` | symmetry-breaking conditions (deterministic kernels) |
| `results/results_symcheck.json` | symmetric-start sanity check (SI ~ 0, CKA = 1) |
| `results/results_phase3.json` | lesion training and gate comparison (30 runs: 6 conditions x 5 seeds) |
| `results/phase3_damage.png` | figure generated from `results_phase3.json` |

## Related work

Not an exhaustive review. Closely related earlier work I am aware of:

- Shevtsova & Reggia, "Lateralization in a bihemispheric neural model," *Neurocomputing* 26 (1999): emergent lateralization in a two-hemisphere letter-identification model, with built-in asymmetries and callosal inhibition.
- Shevtsova & Reggia, "Interhemispheric effects of simulated lesions in a bihemispheric neural model," *Brain and Cognition* 44 (2000), and Reggia, Shkuro & Shevtsova (2001, LNCS 2036): simulated lesions; the intact hemisphere's contribution depends on pre-existing lateralization and lesion size.
- Rajagopalan, Rawlinson, Goldberg & Kowadlo, "Deep learning in a bilateral brain with hemispheric specialization," arXiv:2209.06862: two CNN hemispheres with local/global specialization induced by different training objectives.
- Jacobs & Kosslyn (1994): receptive-field size as a basis for complementary hemispheric representations.

This repository differs in setting (modern deep networks, no assigned roles, narrow bridge and delay as the manipulated variables) but I have not verified that this exact combination is untested. Corrections welcome.

## License

Apache-2.0.
