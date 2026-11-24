# FairDP — Project Evaluation (24 Nov 2026)

How this was evaluated:
- **Code review:** three independent reviews (backend API, frontend dashboard, research core).
- **Browser testing:** scripted end-to-end runs in a real browser (Playwright) across all 11 tabs and the key interactions.
- **Engineering assessments:** API latency measurements, a tech-debt assessment and a test-strategy assessment.
- **Goals audit:** each objective in the master plan checked against evidence.

Every finding marked *fixed* was re-verified after the fix: by an automated test, by the browser run, or both.

## 1. Outcome against the master plan

| # | Objective | Status | Evidence | Main gap |
|---|---|---|---|---|
| 1 | Baseline ML model | **Met** | Credit workstream (LR, MLP, weighted MLP); Adult pipeline: standard + Group DRO | Credit CSV not in the repo, so the credit results cannot be re-run here |
| 2 | DP-SGD under a fixed (ε, δ) | **Met** | `dp/accounting.py`: σ calibrated per target ε, joint gradient+statistics mechanism. Accountant matches Opacus (tests); reviewer confirmed the sensitivity argument | Hyper-parameters chosen on a pilot of the same data (outside the guarantee, now disclosed) |
| 3 | Intersectional fairness analysis | **Met** | 18 sex × race × age groups; TPR/FPR/selection gaps; four-fifths rule; equal opportunity; equalized odds; predictive parity; calibration; individual consistency; differential-fairness ε | No per-group confidence intervals |
| 4 | Mitigation without relaxing privacy | **Partially met** | IG-DPSGD at ε = 1 vs DP-SGD: worst-group recall 41% → 52%; cost of privacy for the worst-hit group 22% → 14% | It miscalibrates across race groups (gap 0.077 vs 0.016), so DP-SGD passes one more scorecard check (8 vs 7). Label balancing reaches 67% worst-group recall. Noisy group statistics (about 3 people per group per batch). Only 5-seed pilots |
| 5 | LLM investigation | **Partially met (L1, L2 done)** | Two Groq models on 200 Adult test people (L1) and 40 people × 8 identities (L2); live audit in the dashboard | L3–L6 (multilingual, benchmarks, leakage, DP-LoRA) not run |
| 6 | Comparative analysis | **Partially met** | Standard vs DP vs mitigated DP, per ε, with membership attack and proxy audit | LLM rows missing; the main grid (3 datasets × 10 seeds) has not been run |
| 7 | Dashboard | **Met** | Next.js + FastAPI; 11 tabs serving real models; end-to-end run with 0 console errors | The end-to-end suite is not yet committed as a test |

**Headline findings, all measured:**
- Every model fails the four-fifths rule for sex.
- The other features reveal sex with AUC 0.93.
- Removing sex and race lowers worst-group recall from 44% to 40% ("fairness through unawareness" fails).
- The three threshold policies show the fairness impossibility trade-off. Demographic parity passes the four-fifths rule but drops worst-group recall to about 43%.

### LLM findings (measured, Groq, zero-shot, temperature 0)
| Model | Accuracy | Balanced acc. | Worst-group recall | Identity flip rate (range) | Positive-rate gap across identities |
|---|---|---|---|---|---|
| qwen/qwen3.8-27b | 73.0% | 62.5% | 25.0% | 0–12.5% (highest: Black woman) | 12.5 pts (Black woman 5% vs White woman 17.5%) |
| openai/gpt-oss-20b | 75.0% | 67.1% | 33.3% | 10–17.5% | 17.5 pts (Asian man 5% vs White man/woman 22.5%) |
| Our IG-DPSGD, ε = 1 (full test set) | 83.8% | 79.6% | 51.8% | identity is a model input, so see the counterfactual grid in the Decision demo | — |

Adding only a race × sex statement changes 10–17.5% of gpt-oss-20b decisions. For Qwen, every identity statement *lowers* the predicted probability, by up to 6.8 pts for "Black woman". Both LLMs are less accurate and less fair to the worst-off group than the privately trained models. They also have no privacy guarantee. (LLM samples are small: 200 and 40 people. Treat the numbers as indicative.)

## 2. Defects found and fixed

| Area | Defect (verified) | Fix |
|---|---|---|
| API, security | A negative `hours-per-week` made the recourse search build a grid of about 2×10⁸ values. The server **hung and used 7.4 GB RAM** | Schema validation on every applicant field (422), plus a bounded recourse grid |
| API, security | LLM `model` id used as a cache path (path traversal on Windows) | Allowlist regex; cache file name sanitised and hashed |
| API | Non-numeric fields, unknown attribute, out-of-range accountant and threshold parameters all returned 500 | Typed and bounded query parameters, 400/422 responses |
| API | Live-training job dict mutated while being serialised (intermittent 500) | Job updates under a lock; responses are deep copies |
| API | The accountant, live-training ε counter and trainer used different step counts | One `_steps()` used everywhere (test added) |
| API | Model card printed ε = 0.000 if the training log was missing; None-formatting crashes | Prints "unknown"; None-safe formatting |
| API | Wildcard CORS | Restricted to the dashboard origin (config) |
| Performance | `/api/sweep` took 3.8 s; `/api/accountant` took 2.3 s per slider move | Lighter sweep + caching: 1.2 s first call, then 14–42 ms; slider requests debounced |
| UI | Stale or out-of-order responses: the slider did not update ε, and old model cards appeared under new names | `useApi` rewritten with request cancellation and reset on path change |
| UI | Mitigation tab: three requests per slider step with no cancellation | Cancellation, cleared results, debounced threshold |
| UI | Double-clicking "Start training" leaked a polling timer | Guard state, single timer, non-overlapping polls |
| UI | Edited applicants still showed "correct/wrong" against the original person's label | Ground truth shown only for the unedited real person |
| UI | Tab switch lost training runs and edits; hidden charts produced about 200 console warnings | Tabs stay mounted off-stage; **0 console warnings** |
| UI | Backend down at load gave "Connecting…" forever | Error panel with retry; automatic reload when the backend returns |
| UI | Remaining hard-coded text ($50k, slider ranges, threshold grid, defaults) | All values now come from `server/config.yaml` through `/api/meta` |
| Research | Per-group membership attack used per-group thresholds | One global threshold; per-group TPR/FPR/advantage (Kulynych et al.) |
| Research | Calibration gap = spread of binned ECE (biased by group size) | Calibration-in-the-large (mean score − actual rate) |
| Research | Undisclosed: tuning on the same data; public group schema; noisy cell statistics; post-processing uses non-private validation data | Added to the model-card limitations |
| LLM | The answer parser mislabelled "Yes"/0–1 probabilities | Parser handles these cases; NaN rejected |
| LLM | Groq rejected every request with 403 (Cloudflare blocks Python's default User-Agent) | Explicit User-Agent header in the client |
| LLM | Hard-coded model id `llama-3.1-8b-instant` had been retired by the provider (404) | Model list fetched live from the provider (`/api/llm`) |
| LLM | Batch run crashed on a transient TLS handshake timeout (only HTTP errors were retried) | Retry with backoff on network errors; the run resumed from the cache |
| LLM | Reasoning models (gpt-oss, Qwen3) wrote chain-of-thought into the answer | Per-model reasoning settings; budget raised to 1024 tokens |
| UI | The LLM model field flashed a text box before the model list loaded | Disabled "loading" dropdown until data arrives |

## 3. Tech debt, prioritised

Priority = (Impact + Risk) × (6 − Effort).

| Item | Type | I | R | E | Priority | Action |
|---|---|---|---|---|---|---|
| All work since the first commit is uncommitted (13 paths) | Infrastructure | 4 | 5 | 1 | **45** | Commit on a branch now |
| End-to-end browser checks not committed as a suite | Test | 3 | 4 | 2 | **28** | Add `@playwright/test` with the scripted flows |
| LLM study has no results | Scope | 5 | 4 | 3 | 27 | Run L1/L2 with a free Groq key on about 300 rows |
| Main grid not run (3 datasets × 10 seeds) | Evidence | 5 | 4 | 3 | 27 | `python experiments/run.py --config experiments/configs/main.yaml` (CPU, hours) |
| No per-group confidence intervals | Code | 3 | 3 | 2 | 24 | Bootstrap per-group TPR over the test set |
| Legacy `src/*.py` and notebooks duplicate the package | Code | 2 | 2 | 1 | 20 | Move to `legacy/` with a note |
| Caches need a server restart after retraining | Architecture | 2 | 2 | 2 | 16 | Key caches on artifact modification time |
| `npm audit`: postcss advisory (build-time) | Dependency | 1 | 2 | 2 | 12 | Upgrade with the next Next.js minor; local-only demo, low exposure |
| `JOBS` grows without bound | Code | 1 | 1 | 1 | 10 | Keep the last N jobs |

## 4. Test status and plan

**Current:** 39 pytest tests, all passing, in five groups:
- privacy accounting vs Opacus
- per-sample gradients vs Opacus
- DP weight and sensitivity invariants
- metrics vs fairlearn, plus the ethics metrics
- API contracts, plus 11 regression tests for the defects above

Scripted browser runs cover every tab and the interactions listed in section 2.

**Gaps, in priority order:**
1. Commit the browser flows as a suite. Smoke-test every tab, plus rapid ε toggling, invalid input, mitigation policy changes, and training with a tab switch.
2. Add a statistical test for the main claim: paired Wilcoxon of IG-DPSGD vs DP-SGD on worst-group recall at 10 seeds, with Holm correction.
3. Add a frontend unit test for `useApi` stale-response handling.
4. Add a load check for the live-training endpoint (one job at a time is enforced).

## 5. Recommended next steps before the evaluation
1. Commit everything (the highest-priority debt item).
2. Run L1/L2 for one free LLM, so the LLM tab shows measured results instead of only the design.
3. Run the main grid for Adult at 10 seeds, overnight on CPU, so claims carry significance tests.
4. In the presentation, show IG-DPSGD as "best balance on the scorecard, not best on every metric". The label-balancing result and the impossibility trade-off are findings, not failures.
