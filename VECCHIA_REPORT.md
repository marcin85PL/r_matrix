# Vecchia prototype: report (16–17 Nov 2026)

Branch `vecchia-prototype` of the toy, based on `golden-export`. The results (JSON) are in
`experiments/vecchia_results/`; notebook 04 reads them.

**Decisions of 17 Nov 2026** (rcorr plan, §6 steps 15 and 16; design note 3 §5):
* block Vecchia is planned as an rcorr stage next to core + halo (a method per class), not scheduled yet;
* the stored-row apply for core + halo (§1, side finding) is planned behind a runtime switch, with the factor path as the
  default until it is timed;
* point Vecchia is rejected.

## 1. Verdict

**The point Vecchia factor of design note 3 (§2.2) is not suitable operationally for our data and decomposition.
A block variant is: it keeps all the benefits, at about the cost of core + halo and with half its communication.**

| | core + halo (today) | point Vecchia (design note 3) | **block Vecchia (new)** |
|---|---|---|---|
| accuracy at the operational setting | reference | matches core + halo only at m ≈ 100 | **within 2× of core + halo at the same h (sometimes better)** |
| accuracy for the same bytes | reference | worse | **16–64× better (h = 6c)** |
| SPD | after symmetrisation, checked | by construction | **by construction** |
| R⁻¹, R^−½, R^½ from one operator | R⁻¹ only | yes | **yes** |
| halo / core, 96 → 1024 tasks | 2.2 → 11.4 | 7 → 33 | **1.1 → 5.7** |
| bytes per apply, busiest task, 1024 tasks | 8.3e4 | 1.5e5 | **8.0e4 (colour), 4.8e4 (id)** |
| peers per task, 1024 tasks | 35 | **875 (almost all-to-all)** | 33 (colour), 22 (id) |
| apply flops, busiest task, 1024 tasks | 3.2e6 as rcorr does it (Cholesky solves); **1.6e5 with stored core rows** | 0.9e5 | 1.5e5 (colour), 1.1e5 (id) |
| setup flops, busiest task, 1024 tasks | 4.1e7 (5.4e7 with stored rows) | 7.7e7, plus **6 search rounds** and a global ordering | 4.3e7 (colour), 1.6e7 (id) |
| sampling rounds (W⁻¹z), 1024 tasks | no R^½ | **≈ 1200** | **15 (colour)**, 264 (id) |
| reproducible (D2) | yes | yes | yes |

* **Block Vecchia.** Each R-patch conditions jointly on the data of *earlier* patches within h: a **one-sided halo**.
  * **Accuracy:** within a factor of 2 of the symmetrised core + halo at the same h, sometimes better, on all 9 golden cases and on real SEVIRI images (ch5, ch6, ch9) from 96 to 1024 tasks. A slightly wider h (about 4.5–5c) would match it and still costs less communication.
  * **Communication:** half that of core + halo.
  * **Apply flops:** about the same as a core + halo that stores its core rows (1.0–1.9× fewer).
  * **Accuracy at equal communication:** h = 6c costs fewer bytes than core + halo at 4c and is 16–64× more accurate.
  * **What it adds:** SPD by construction, R⁻¹, R^−½ and R^½ from one factor, and samples whose covariance is exactly C̃.
  * **The patch sequence is a trade-off.** In the colour sequence the last colours condition on all their neighbours: few sampling rounds, but uneven load. The patch-id sequence balances the load and lowers bytes, peers and setup, at the price of many sampling rounds.
* **Point Vecchia.** Its m nearest earlier neighbours reach far beyond the correlation support: median 900 km, 90th percentile 2700 km at m = 100, against c = 115 km. The reason is that SEVIRI data are thinned to a spacing of about c. In the max-min ordering the coarse data of every patch condition on data all over the disc. The consequences:
  * the communication is **almost all-to-all**;
  * the neighbour search needs 6 rounds; only 52 % of the data are certified after 2;
  * sampling needs about 1000 sequential rounds.

  Ordering patch by patch (D2-friendly) helps at 96 tasks: halo / core 2–7, peers 9–16, about 50 rounds, and m = 100 matches core + halo. It loses again at 1024 tasks: halo / core 11–17 (1.5× core + halo), peers 31–38, about 340 rounds. The neighbour search still needs 6 rounds. Apply flops are the lowest of the three, its only clear advantage.
* **A side finding for today's rcorr.** The busiest-task apply of core + halo scales poorly, ×4.7 for ×10.7 tasks. The reason is that it solves with the Cholesky factor of core + halo (4 n² flops, n = core + halo). If the core rows of R_loc⁻¹ are stored instead (4 n_core n flops; memory n_core·n, *less* than the factor at large P), the apply at 1024 tasks becomes **20× cheaper** (3.2e6 → 1.6e5 flops). The setup grows by 30 % at 1024 tasks (×2.9 at 96). The numbers are those of core + halo; nothing else changes.

**Recommendation.**
* Keep point Vecchia as documented but rejected.
* Take block Vecchia as the candidate "factor" method for rcorr. It reuses almost everything that exists: the partition, halo discovery (pushed one way), comm maps, the adjoint halo exchange, and per-block LAPACK.
* It would give:
  * R^½ for the EDA directly, as an alternative to Chebyshev on mult;
  * R^−½ for the minimisers that abort today;
  * an R⁻¹ with half the communication of core + halo and about its accuracy.
* The next steps are in §6.

## 2. What was built

`r_matrix/vecchia.py`, about 600 lines; tests in `tests/test_vecchia.py`, 21 tests; the whole toy suite (131 tests) passes.

* **Serial references.**
  * Orderings: max-min (`maxmin_order`, a function of the data, ties by global ID) and patch-wise max-min.
  * `patch_sequence`, with the patches in `id` order or by greedy `colour` of the patch graph.
  * `point_W` and `block_W`, with exactness checks.
* **Distributed setups on `MockComm`.** Both builders produce, per patch, the owned rows of W as a CSR matrix over [own data; halo].
  * `setup_point`: certified neighbour search by radius-doubling position pushes, with optional replication of the coarse data. Certified selections are bit-identical to the serial ones.
  * `setup_block`: one push of positions from earlier to later patches within h, then per group one dense factorisation of [N; B]. W_BB = L⁻¹ and W_BN = −L⁻¹S are stored.
* **Recurring operations**, generic for both variants:
  * `apply_W` (one halo exchange);
  * `apply_Wt` (local transpose, then the toy's existing adjoint halo exchange);
  * `apply_inv` = WᵀW;
  * `sample` = W⁻¹z by forward substitution in rounds. Only new values travel, and every round is counted.
* **Evaluation.**
  * `r_matrix/vecchia_eval.py`: golden-case data, the dense core + halo operator for the same data, and cost counters.
  * `experiments/vecchia_golden.py`, `vecchia_seviri.py` and `vecchia_scaling.py`.
* **Notebooks.**
  * `notebooks/03_vecchia_prototype.ipynb`: the implementation, step by step, on the geo-disc golden case.
  * `notebooks/04_vecchia_evaluation.ipynb`: all results; it reads the JSON files in `experiments/vecchia_results/`.
  * The generators are `experiments/make_nb03.py` and `make_nb04.py`.

**Tests**, against independent references:
* W with complete conditioning sets reproduces C⁻¹ to 1e-12, i.e. it is the exact inverse Cholesky factor.
* The distributed W equals the serial W: point bit for bit, block to 1e-12.
* WᵀW through the halo exchanges equals the global product.
* ⟨y, Ax⟩ = ⟨x, Ay⟩, and W and Wᵀ are adjoint through the exchanges.
* SPD for any m, including clustered data at α 0.9.
* C̃⁻¹x and the samples are bit-identical on 1, 2, 4 and 7 tasks (8 fixed patches).
* Sampling solves Wx = z.
* Groups and classes are independent; groups smaller than m are exact; poles and dateline need nothing special.

## 3. How it was evaluated

* **Golden cases.** All 9 cases of `export.CASES`, per group, with 4 and 8 RCB patches, each at the class's halo width h. 1.5c to 4c; c of 400 to 1500 km.
* **Real data.** All active Met-11 SEVIRI data of the jdj6 cycle 2022-06-01 00 UTC: 203 633 data in 36 groups (channel × 15-min slot), with d = fg_depar / σ_o and c = 115 km.
  * The patches are the RCB partition of **all** data of the cycle into P = 96, 384 or 1024 patches, as rcorr does.
  * Single images: ch6 slot 28 (n = 5728) at α 0.5, 0.7 and 0.9; ch6 slot 8; ch9 slot 28 (n = 924); ch5 slot 28 (n = 11 415).
  * Every operator is compared with the exact dense C.
* **Metrics.**
  * Accuracy: the generalised eigenvalues of C̃⁻¹C (max |λ − 1| as on slide 16, λ_min, KL divergence), the relative error of Jₒ = dᵀC̃⁻¹d and of C̃⁻¹d for the real departures.
  * Implied covariance: the errors of the implied variances diag(C̃), which EDA perturbations would have, and of all its entries.
  * Costs: flops of setup and apply on the busiest patch; bytes, messages and peers counted by MockComm; setup rounds; sampling rounds.
* **Scaling.** The same cycle, all 36 groups, 96 to 1024 tasks, counted structurally per task, summed over groups (no factorisation).

All costs are **flop and message counts, not timings**. The toy is serial Python.

## 4. Results

### 4.1 Accuracy

**Golden cases** (8 patches, max |λ − 1|, the case's h):

| case | core + halo | block (colour) | point m = 20 | point m = 50 |
|---|---|---|---|---|
| sph_uniform | 1e-2 | 9e-3 | 2e-2 | 4e-4 |
| sph_clustered (α 0.9) | 1e-2 | 3e-3 | 4e-1 | 8e-2 |
| sph_geo_disc | 7e-2 | 3e-2 | 9e-1 | 3e-1 |
| sph_poles_dateline | 1e-2 | 2e-2 | 4e-2 | 3e-3 |
| sph_inactive | 4e-3 | 7e-3 | 4e-4 | 1e-7 |
| sph_short_halo (h = 1.5c) | 6e-2 | 1e-1 | 2e-3 | 5e-6 |
| groups / classes / tiny | ≤ 1e-3 | ≤ 1e-3 | exact (n ≤ m) | exact |

Block tracks core + halo everywhere: its error is 0.3–2× that of core + halo. Point Vecchia is excellent where the data are sparse relative to c. Where many data lie within c (clustered, geo disc) it is poor: λ_min drops to about 0.5 at m = 20, though it stays SPD.

**Real SEVIRI ch6, slot 28** (max |λ − 1|):

| α, P | core + halo h = 4c | block h = 4c | block h = 6c | point m = 50 / 100 (max-min) |
|---|---|---|---|---|
| 0.5, 96 | 2.5e-4 | 2.2e-4 | 3.9e-6 | 3.7e-3 / 1.8e-4 |
| 0.7, 96 | 1.1e-3 | 1.5e-3 | 4.7e-5 | 1.3e-2 / 1.1e-3 |
| 0.9, 96 | 1.2e-2 | 1.1e-2 | 8.8e-4 | 5.4e-2 / – |
| 0.7, 384 | 1.2e-3 | 1.8e-3 | 4.2e-5 | 1.3e-2 / 1.1e-3 |
| 0.5, 1024 | 3.7e-4 | 4.6e-4 | 1.0e-5 | (see notebook) |
| 0.7, 1024 | 1.6e-3 | 1.8e-3 | 1.0e-4 | (see notebook) |

**ch9 slot 28** (n = 924, α 0.5):
* 96 patches: core + halo 1.2e-4, block 1.9e-4, block h = 6c 1.3e-6, point m = 100 1.3e-4.
* 1024 patches: core + halo 2.4e-4, block 1.5e-4.

**ch5 slot 28** (n = 11 415, the densest channel):

| α, P | core + halo h = 4c | block h = 4c (colour / id) | block h = 6c | point m = 100 (max-min / patch) |
|---|---|---|---|---|
| 0.5, 96 | 3.2e-4 | 6.3e-4 / 6.3e-4 | 1.5e-5 | 3.8e-4 / 4.4e-4 |
| 0.7, 96 | 1.4e-3 | 2.2e-3 / 2.2e-3 | 8.0e-5 | 1.7e-3 / 2.2e-3 |
| 0.7, 1024 | 2.4e-3 | 2.5e-3 / 2.9e-3 | 1.2e-4 | 1.7e-3 / 1.2e-3 |

**ch6 slot 8** (another image, n = 5871, α 0.7, 96 patches): core + halo 1.2e-3, block 1.6e-3 (colour) and 1.3e-3 (id), block h = 6c 5.7e-5.

* **Accuracy depends on h and α, hardly on P.** At h = 4c, block is 0.6–2× the core + halo error. Both degrade with α the same way: at α 0.9, h = 4c gives about 1e-2 for both. The patch sequence (colour or id) does not change the accuracy.
* **Jₒ:** the relative error of dᵀC̃⁻¹d for the real departures is ≤ 1.5e-5 for core + halo and block at h = 4c, up to α 0.9. The relative error of C̃⁻¹d is ≤ 2.6e-4 for α ≤ 0.7 and about 1e-3 at α 0.9, similar for block and core + halo.
* **Implied variances:** block is closer to 1 than core + halo, by 3–11×. At α 0.7 and P = 96: 2.5e-7 against 8.3e-7; at α 0.9: 9e-6 against 1e-4.

### 4.2 Accuracy for the communication it costs

ch6, 96 patches, bytes of one application:

| | core + halo h = 4c | block h = 4c | block h = 6c |
|---|---|---|---|
| bytes | 2.0e5 | 1.0e5 | 1.8e5 |
| max \|λ − 1\| at α 0.5 | 2.5e-4 | 2.2e-4 | 3.9e-6 |
| max \|λ − 1\| at α 0.7 | 1.1e-3 | 1.5e-3 | 4.7e-5 |

At 1024 patches, block h = 6c also costs fewer bytes than core + halo at 4c (1.0e6 against 1.06e6) and is 16–37× more accurate. Notebook 04, section 2.1, plots accuracy against bytes, apply flops and setup flops: block dominates core + halo in the first two, and point Vecchia is worst in bytes.

### 4.3 Symmetry and positive definiteness

* **Vecchia:** C̃⁻¹ = WᵀW is symmetric to rounding through the distributed exchanges (|⟨y,Ax⟩ − ⟨x,Ay⟩| ≈ 1e-13 relative). It is SPD in every case tested, all m and h, golden and real, including m = 1 on clustered data. No symmetrisation step is needed.
* **Core + halo without symmetrisation:** asymmetric at 1e-4 relative (ch9).

### 4.4 Cost per task, all 36 groups of the cycle (scaling)

| tasks | | core + halo | core + halo, stored rows | block h = 4c, colour | block h = 4c, id | point m = 100 |
|---|---|---|---|---|---|---|
| 96 | halo / core | 2.21 | 2.21 | 1.09 | 1.10 | 7.1 |
| | apply flops, busiest task | 1.5e7 | 4.3e6 | 3.0e6 | 2.3e6 | 8.6e5 |
| | setup flops, busiest task | 6.0e8 | 1.8e9 | 8.7e8 | 4.5e8 | 7.5e8 |
| | bytes per apply, busiest task | 1.2e5 | 1.2e5 | 1.1e5 | 7.3e4 | 3.6e5 |
| | peers, busiest task | 10 | 10 | 10 | 6 | 95 |
| | sampling rounds | – | – | 5 | 39 | 953 |
| 1024 | halo / core | 11.4 | 11.4 | 5.7 | 5.7 | 33 |
| | apply flops, busiest task | 3.2e6 | 1.6e5 | 1.5e5 | 1.1e5 | 8.8e4 |
| | setup flops, busiest task | 4.1e7 | 5.4e7 | 4.3e7 | 1.6e7 | 7.7e7 |
| | bytes per apply, busiest task | 8.3e4 | 8.3e4 | 8.0e4 | 4.8e4 | 1.5e5 |
| | peers, busiest task | 35 | 35 | 33 | 22 | 875 |
| | sampling rounds | – | – | 15 | 264 | 1220 |

All 36 groups of the cycle, per task (= patch), summed over its groups. Bytes are 8 per halo value, twice (halo and adjoint halo). Notebook 04, section 3, has all of 96, 192, 384, 768 and 1024.

* **Communication.**
  * Block: half the halo of core + halo at every P, with the same peers (colour) or fewer (id: 22 against 35 at 1024 tasks).
  * Point: halo / core of 7 (96 tasks) to 33 (1024 tasks), and **the busiest task talks to 95 of 96, or 875 of 1024 tasks**. The coarse data of every patch have neighbours all over the disc.
* **Apply flops of the busiest task.**
  * Core + halo as rcorr does it scales poorly (×4.7 for ×10.7 tasks): the n² of the halo dominates.
  * Stored core rows and block Vecchia both scale super-linearly (×28 and ×20): they cost n_core·n, and n_core shrinks.
  * Block against stored-row core + halo: 1.4× fewer flops at 96 tasks, 1.05× at 1024 (colour); 1.9× and 1.4× (id).
* **Setup flops of the busiest task.**
  * Block with the colour sequence: 1.0–1.4× core + halo. The last colour conditions on all its neighbours, so it is the busiest.
  * Block with the patch-id sequence: 0.3–0.75×.
  * Stored-row core + halo: 1.3–2.9×.
  * The setup runs once per minimisation.
* **Memory, busiest task** (8-byte words, 1024 tasks): core + halo factor 4.0e5; stored rows 3.9e4; block 3.7e4 (colour) or 2.7e4 (id); point 4.4e4.
* **Setup of point Vecchia.** 6 radius-doubling rounds from R0 = 4c. The fraction of data certified after 1, 2, 3, 4 rounds is 0.8, 52, 82, 94 %. The global max-min ordering is sequential in the toy: it is computed centrally, and a parallel version is an open problem. The patch-wise ordering avoids that.

### 4.5 Sampling (R^½ for the EDA)

* x = W⁻¹z has covariance exactly C̃; the Monte-Carlo check is in notebook 03.
* Rounds are the sequential communication steps of the forward substitution:
  * **block, colour:** 4–6 at 96, 8–14 at 384, 14–23 at 1024 patches;
  * **point, max-min:** 160–1220;
  * **point, patch order:** 40–50 at 96 patches, 240–340 at 1024.
* Each round is one sparse neighbour exchange of new values only. At 1024 tasks block sampling has the message count of 7–8 applications of R⁻¹ (two exchanges each), and the flops of one product with W.
* The alternative, a Chebyshev polynomial on the exact mult, needs 7–15 products for 1e-3 to 1e-4. Its samples have a covariance close to C, not exactly the C̃ that 4D-Var inverts.

## 5. Operational suitability: what to watch

* **Block Vecchia.**
  * **Patch sequence and load balance:** colour gives few sampling rounds but makes the last colours the busiest (setup up to 1.4× core + halo, bytes and peers like core + halo). Id balances the load and lowers bytes and peers by about a third, but sampling then needs 40–260 rounds. Worth trying: a sequence with few colours *and* balanced one-sided halos, for example alternating the order at every level of the RCB tree. Both sequences give the same accuracy.
  * **Per-class halo width:** each class needs its own h; the toy uses one h per build. Straightforward in rcorr, which already has per-class h.
  * **Dependence on the layout:** like core + halo, the operator depends on the patch layout. The reproducible mode (fixed patches) covers D2.
  * **Choice of h:** h = 4c is within 2× of today's accuracy, and h = 5–6c buys one to two orders of magnitude for about the communication of today's core + halo.
  * **Dense or clustered data:** fine at α 0.9 (golden sph_clustered 3e-3).
* **Point Vecchia.**
  * **Reach:** set by the data density (spacing ≈ c), not by the correlation length. Capping at 2ℓ stalls the accuracy (the 16 Nov test).
  * **Choice of m:** m depends strongly on α and on the data density; it would have to be monitored per class.
  * **Ordering:** the global ordering is sequential.
  * **Conclusion:** on a domain decomposition with SEVIRI-like density it loses on every count except apply flops.
* **Not evaluated.**
  * Wall-clock timings and latency: the toy is serial.
  * Impact on the analysis and forecasts.
  * Correlations across slots or channels.
  * Denser instruments: FCI, or SEVIRI unthinned.
  * One cycle and one date only; one slot-8 image as a check.

## 6. Next steps (not started)

1. **Done:** design note 3 §2.2, §3 and §5 and the plan (§6 steps 15 and 16) updated on branch `vecchia-note` of rcorr.
2. **Toy:**
   * a block mode with per-class h;
   * a balanced patch sequence (few colours, even one-sided halos);
   * the stored-row apply for core + halo (an easy gain for rcorr today);
   * a comparison with Chebyshev R^½ for the EDA: cost and sample covariance against C̃.
3. **rcorr stage (planned, plan §6 step 15):** a block Vecchia method next to core + halo, chosen per class.
   * **Reused:** partition, halo discovery with a one-way push filter, comm maps, adjoint halo exchange, per-(patch, class) `dpotrf`.
   * **New:**
     * a patch sequence (colouring) as a function of the layout;
     * stored W blocks (`dtrmv`/`dgemv`) or triangular solves;
     * `rcorr_inv_sqrt` (W) and `rcorr_sqrt` (forward substitution in rounds);
     * golden data from this prototype.
4. **Real-system check:** C̃⁻¹ against core + halo in the IFS, with the round-trip and Jₒ checks as for core + halo, then one cycling experiment.

## 7. Files

| what | where |
|---|---|
| code, tests | `r_matrix/vecchia.py`, `r_matrix/vecchia_eval.py`, `tests/test_vecchia.py` |
| experiments | `experiments/vecchia_golden.py`, `vecchia_seviri.py`, `vecchia_scaling.py` |
| notebooks | `notebooks/03_vecchia_prototype.ipynb`, `notebooks/04_vecchia_evaluation.ipynb` (generators `experiments/make_nb0[34].py`) |
| results (JSON) | `experiments/vecchia_results/` (`golden.json`, `acc_ch*_s*_a*_P*.json`, `scaling.json`) |
| input data | `jdj6_20220601_00.npz`, from the jdj6 observation feedback (`ofb_geos_4023_active_20220601_0012.odb`); ECMWF data, not in the repository |
