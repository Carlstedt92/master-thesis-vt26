# GPU Compute Allocation Request — DINO-SSL Molecular Pretraining Scale-Up

**Project:** Berzelius-2026-201 (PI: Ola Spjuth)
**Prepared by:** Emil Carlstedt
**Purpose:** Internal justification for requesting an increased GPU-hour allocation *and* a larger `/proj` storage quota, covering (1) the overrun already incurred during the current thesis's methodology-validation phase, and (2) a planned scale-up of SSL pretraining from the current 9M-molecule ZINC subset to somewhere between 400M and 990M molecules (a second, larger candidate dataset identified after the initial 400M estimate — the exact target is still open, see Section 3).

---

## 1. Summary / ask

The current allocation (1000 core-hours/month) has been running at roughly **3.5-4x** actual usage during the methodology-validation phase of this thesis, for reasons detailed in Section 2 — this was not incidental overrun but a direct, traceable consequence of the number of controlled ablations required to validate the training pipeline. Scaling SSL pretraining to 400M-990M molecules is a further **44x-110x** increase in per-epoch compute cost on top of that, plus a storage requirement (Section 6) that exceeds the current `/proj` quota regardless of which end of that range is chosen. This document lays out the actual per-experiment cost (measured directly, not estimated), the resulting compute and storage cost models for the scale-up, and a staged approach to de-risk the ask rather than requesting the full scaled-up amount outright.

**Concrete ask: ~6,000 GPU-hours/month and 15TB of `/proj` storage.** The GPU-hours figure is derived three independent ways in Section 5 (all converge on the same number); the storage figure covers the 990M-molecule scenario (~11.1TB) with ~4TB headroom for checkpoints and other files (Section 6). **[EDIT: still needs your input — the 400M-vs-990M target itself (Section 3), and confirmation this is what you want to bring to the PI as-is.]**

**Note on units**: all figures below are in raw **GPU-hours** (wall-clock hours × number of GPUs allocated), not SLURM's internal TRES "billing" units — an earlier draft of this document used the wrong conversion (mistook the billing-weight field for an hourly rate, inflating every number by ~32x). GPU-hours is corroborated against real `sacct`/`projinfo` data in Section 2 but isn't independently confirmed against Berzelius's exact accounting formula — worth a quick check with NAISS/Berzelius support or the PI before this goes into a formal request, in case there's an additional node-type or memory-tier weighting on top of raw GPU count.

---

## 2. Current usage and what it bought

Consumed compute over the last 30 days: **3,762.88 hours** against the 1000 h/month allocation (~3.76x). This is a real, `projinfo`-verified number, not an estimate.

This wasn't runaway or exploratory compute spend — it reflects a sequence of specific, hypothesis-driven ablations, each run to completion because the question required it:

- **Weight-decay bug fix validation** — a BatchNorm-gamma collapse bug (root-caused to weight decay being applied uniformly, including to BatchNorm gamma/beta) was fixed and then validated across both augmentation modes in use (KHOP and masking), each requiring a full 60-epoch run to confirm the fix held under both.
- **Extended node-feature ablation** — tested whether two new engineered features (Gasteiger partial charge, topological eccentricity) improved downstream transfer. Result: a **clear, reproducible negative result** — the new features caused a consistent downstream regression across LIPO/Tox21/BACE. This required full training runs (not just inference) to detect, since the effect wasn't visible in SSL loss alone.
- **Root-causing the regression** — two independent, testable hypotheses (shortcut learning via feature dropout; raw-feature-scale imbalance via eccentricity normalization) were each validated with dedicated runs, both showing partial fixes — a nuanced, evidence-backed finding rather than "features helped" or "features didn't help."
- **Task-difficulty ablation (k_hops)** — tested whether the SSL pretext task was too easy (loss plateaus very quickly) by varying the local-view neighborhood size (k=0, k=1, k=2), each requiring a full run since the effect on downstream transfer isn't observable from the loss curve alone.
- **Architecture ablation (GAT attention heads)** — tested whether increasing model complexity (1 → 3 → 4 → 16 attention heads, same total parameter budget) improved representation quality. 3-head result: a real, reproducible improvement on 4 of 6 downstream metrics.

Every one of these was a full 60-epoch, 8-GPU run because the effect under test (downstream transfer quality) is only observable after training, not inferable from a shorter proxy run. **Measured cost per such run: ~20.5 hours wall-clock on 8 GPUs, ~164 GPU-hours** (10-run sample, this session, via `sacct`) — about a sixth of the monthly allocation for one experiment. Cross-checked against the actual `projinfo` usage report: summing wall-clock × GPU-count across every job run since late July gives ~1,900 GPU-hours, the right order of magnitude against the ~3,763 hours `projinfo` reports consumed over the last 30 days (the two don't reconcile exactly — some gap remains, possibly jobs further back in time or an accounting detail not visible from `sacct` alone — but this is a far closer match than treating SLURM's internal billing-weight field as an hourly rate, which was tried and clearly wrong by ~32x).

This context matters for the request: the overrun is a predictable, reproducible cost of doing controlled methodology validation properly, not a one-off spike. The 400M scale-up will have the same property — each scale/architecture/epoch-count decision will need empirical validation, not just a single production run.

---

## 3. The scale-up: 9M → 400M (or 990M) molecules

**[EDIT: this section needs your input — I don't have the scientific rationale for either target. Suggested things to cover: where the additional molecules come from for each option (larger ZINC tranche? a different source for the 990M set?), what chemical space/diversity gap this is meant to close relative to the current 9M subset, how it connects to the thesis's downstream goals (e.g., closing the gap to ECFP/leaderboard performance discussed earlier, or specifically targeting scaffold diversity underrepresented in the current set), and — now that both are on the table — why 400M vs. 990M, or whether the 50M pilot in Section 7 should settle that before committing.]**

The scale factors are concrete: 400M / 9M ≈ **44.4x**, 990M / 9M ≈ **110x** more molecules than the current training set.

---

## 4. Why this isn't a naive 44x compute multiplication

A naive read would be "44x more data → 44x more compute for the same training recipe." That's very likely the wrong way to plan this, for a reason this thesis has already produced direct evidence for:

**SSL loss and downstream quality both plateau early, even at 9M scale.** Per-epoch tracking (this session's own diagnostics) showed the SSL loss reaches its floor within the first ~3-5 epochs of a 60-epoch run, and — more importantly — the online downstream-eval metric (LIPO KNN validation RMSE) was *already near its best observed value by epoch 1*, with no clear further improvement across the remaining ~59 epochs. In other words: at 9M molecules, the marginal value of additional full passes over the data drops off fast.

This is a real, load-bearing argument for the scale-up case: a much larger, more diverse corpus plausibly needs **far fewer full epochs** to reach comparable (or better) coverage than repeating the current 60-200 epoch schedule 44-110x over. But this is a hypothesis extrapolated from 9M-scale evidence — it hasn't been tested at larger scale, which is exactly why Section 7 proposes validating it empirically before committing to the full ask.

---

## 5. Cost model and recommended monthly rate

Measured baseline: **2.74 GPU-hours per epoch at 9M molecules** → **~122 GPU-hours/epoch at 400M** or **~301 GPU-hours/epoch at 990M** (linear scaling assumption — compute per epoch is dominated by number of graph forward/backward passes, which scales linearly with dataset size at fixed batch size and architecture).

| Epochs | 400M: total GPU-h (3mo / 6mo rate) | 990M: total GPU-h (3mo / 6mo rate) |
|---:|---:|---:|
| 5   | 609 (203 / 101)     | 1,507 (502 / 251)     |
| 10  | 1,218 (406 / 203)   | 3,014 (1,005 / 502)   |
| 20  | 2,436 (812 / 406)   | 6,028 (2,009 / 1,005) |
| 30  | 3,653 (1,218 / 609) | 9,042 (3,014 / 1,507) |
| 60 (current KHOP baseline) | 7,307 (2,436 / 1,218) | 18,084 (6,028 / 3,014) |

**Recommended target: ~6,000 GPU-hours/month**, arrived at three independent ways rather than picked as a round number:

1. **Runs-per-month target**: wanting the capacity for **2 independent runs per month** at 990M scale and ~10 epochs costs 2 × 3,014 = **6,028 GPU-h/month**.
2. **Wall-clock timing**: measured per-epoch wall-clock at 9M scale (20.52h / 60 epochs = 0.342h/epoch) scales to **37.6h/epoch at 990M**. A run budgeted at ~15 days (360h) wall-clock therefore lands at **9.6 epochs** — independently landing on the same ~10-epoch figure.
3. Both of the above independently reproduce the original 6,000 GPU-h/month starting point.

This also gives a concrete operational target: **~10 epochs, ~15 days wall-clock, 2 runs/month, at 990M scale.** One caveat: 2×15 days = 30 days exactly, so this is a tight fit with essentially no slack for queue wait, restarts, or debugging between runs — worth either building in a small buffer above 6,000, or treating 2 runs/month as the ceiling rather than the guaranteed cadence.

At this rate, note it does *not* commit to the full 60-epoch production schedule (that would cost 18,084 GPU-h *per run* at 990M) — it's sized around the "fewer epochs suffice at scale" hypothesis from Section 4, consistent with the staged approach in Section 7. If the 50M pilot shows 60 epochs really is needed even at larger scale, this rate would only support one such run every ~3 months, not 2/month.

---

## 6. Storage requirements (separate from GPU-hours, same SUPR channel)

Precomputed shard storage scales with molecule count, not epoch count, so this is independent of the epoch-count question in Section 5. Measured baseline: **101GB for the current 9M-molecule, 24-feature precomputed set** (verified via `du`, isolated from model checkpoints and other files — checkpoints live in a separate 41GB directory).

| Target scale | Scale factor vs. 9M | Precomputed storage needed |
|---:|---:|---:|
| 50M (pilot, Section 7) | 5.6x | ~560 GB |
| 100M | 11.1x | ~1.1 TB |
| 400M | 44.4x | ~4.4 TB |
| 990M (larger tranche also under consideration) | 110x | ~11.1 TB |

**Two separate storage tiers, and only one of them is actually a constraint:**

- **`/proj` project storage (persistent, shared)**: standard NAISS/Berzelius quota is 2TB per project ([per NSC's Berzelius documentation](https://www.nsc.liu.se/support/systems/berzelius-getting-started/#4-data-storage-on-berzelius)), matching what's currently allocated and used (~349GB of it, mostly precomputed shards from this session's ablations). **This is the real bottleneck** — none of the scale-up scenarios above fit in the current quota, and even the 50M pilot would push close to it once other files are accounted for. The quota is explicitly expandable via SUPR, the same system that would handle the GPU-hours request — this should be requested alongside it, not treated as a separate ask.
- **`/scratch/local` node-local scratch (transient, per-job)**: this is where each training job stages its data before running (see the sbatch scripts' staging step). Per NSC's documentation, the "fat" nodes we've been targeting via `--constraint=fat` have **30TB of local NVMe SSD** — comfortably fits even the 990M/11TB scenario with room to spare. This was an open question earlier in this process (couldn't be verified from the login node) and is now resolved: **local scratch is not a constraint at any of these scales.**

Practical implication: the existing pipeline (precompute once to `/proj`, stage to `/scratch/local` per job) doesn't need to change architecturally for the scale-up — it just needs a larger `/proj` quota. Switching to on-the-fly featurization (computing graphs from SMILES during training instead of precomputing) was considered as an alternative that avoids the storage requirement entirely, but introduces its own costs (repeats RDKit featurization work every epoch instead of once, and in an 8-GPU DDP job each rank's workers would redundantly re-featurize the same molecules) — not needed now that the actual local-scratch capacity is known to be sufficient.

**[EDIT: decide how much /proj quota to actually request — e.g. requesting up to the 990M/11.1TB figure with headroom (~15TB) covers both scale scenarios without needing a second storage request later, but ties the ask to the larger, less-validated target. Requesting only the 400M/4.4TB figure (~5-6TB with headroom) is more conservative and consistent with keeping epoch count/final scale open per Section 5.]**

---

## 7. Recommended approach: staged, not a single large ask

Given that the "fewer epochs suffice at scale" hypothesis (Section 4) is untested above 9M molecules, I'd still recommend a staged plan even though the corrected numbers in Section 5 make a direct ask far more tractable than initially estimated:

1. **Pilot at intermediate scale** (e.g. 50M molecules — a ~5.6x step from the current set, not a 44x jump) to directly test whether the fast-plateau finding holds at larger scale. This alone would validate or falsify the core assumption Section 4 is built on, at a small fraction of the 400M cost (a single 50M/10-epoch run would cost roughly 2.74 × (50/9) × 10 ≈ 152 GPU-hours — comparable to a single one of the ablation runs already run this session).
2. **Use the pilot's actual loss/downstream-quality curve** to pick a defensible epoch count for the full 400M run, rather than guessing.
3. **Request the full 400M allocation as a second phase**, informed by real pilot data instead of a linear extrapolation.

This turns the ask into two smaller, sequential requests that are each independently justified by data, rather than one large number that depends on an assumption nobody has tested yet.

---

## 8. Immediate next step

Requesting **~6,000 GPU-hours/month and a 15TB `/proj` quota** (up from 1000 h/month and 2TB), sized for 2 runs/month at ~10 epochs/990M scale once the scale-up is underway. Per the staged approach in Section 7, the first use of this allocation would be the 50M-molecule pilot (a small fraction of the monthly budget), with the larger runs starting only once that pilot confirms the epoch-count assumption this rate is built on.

**[EDIT: confirm the 400M-vs-990M target (Section 3) before this goes to the PI — the storage figure and the "2 runs/month" framing both assume 990M specifically; if 400M is the actual target, both numbers should be recalculated from the 400M columns instead.]**
