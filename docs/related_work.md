# Related work & novelty check (living document)

Status: initial screen done 2026-11-24 (web search, 3 queries). **Each entry
must be read in full and verified by a team member before citing.** Items
marked (verify) were seen only as search snippets.

## 1. Gap table

| Work | Private? | Worst-group / group-aware | Intersectional | Sparse-cell handling | Budget split for group stats | Notes |
|---|---|---|---|---|---|---|
| Bagdasaryan et al. 2019, *DP has disparate impact on model accuracy* | ✓ | eval only | – | – | – | Origin of problem |
| Xu, Du, Wu 2021, *DPSGD-F* (KDD) | ✓ | group clipping | – | – | implicit | Baseline `dpsgd_f` (approx.) |
| Esipova et al. 2023, *Disparate impact from gradient misalignment* (ICLR) | ✓ | global scaling | – | – | – | Baseline `global_adapt` |
| Tran, Fioretto et al. 2021 (AAAI; NeurIPS) | ✓ | Lagrangian / fairness lens | – | – | – | Theory of clipping-induced unfairness |
| Jagielski et al. 2019; Lowy et al. 2023 (DP-FERMI) | ✓ | fairness constraint | – | – | – | Private sensitive attrs |
| Noe et al. 2022, *Unfairness of DP-SGD across settings* (arXiv 2202.12058) | ✓ | DP-SGD + GroupDRO (Opacus) | – | – | – | Finds privacy–fairness negative correlation |
| (Bagdasaryan-style) 2022/24, *Impact of DP on group disparity mitigation* (Findings NAACL 2024, arXiv 2203.02745) | ✓ | GroupDRO | – | – | – | DP can *improve* fairness under DRO – conflicts with Noe |
| Zhou & Bassily 2024, *DP worst-group risk minimization* (arXiv 2402.19437) (verify authors) | ✓ | ✓ theory | – | – | – | Formal rates; not practical deep learning |
| *Adaptive sampling and clipping for private worst-case group optimization* (arXiv 2602.10820, Feb 2026) (verify) | ✓ | ✓ sampling + clipping | ? | ? | ? | **Closest prior work – must read fully.** |
| *A simple and practical method for reducing DP disparate impact* (arXiv 2312.11712) (verify) | ✓ | ✓ | ? | – | – | |
| *Revisiting disparate impact of DP-SGD* (arXiv 2510.01744) (verify) | ✓ | eval | – | – | – | Disparate impact largely from reusing non-private hyper-params → we tune DP baselines separately |
| *Group privacy-risk disparity, DP-SGD-S* (arXiv 2510.09114) (verify) | ✓ | group clipping for **privacy** fairness | – | – | – | Motivates our per-group MIA |
| Berrada et al. 2023, *Unlocking accuracy & fairness in DP image classification* (arXiv 2308.10888) | ✓ | eval | eval only | notes small-subgroup failure | – | |
| Foulds et al. 2020, *Intersectional definition of fairness* | – | – | ✓ (ε_DF) | Bayesian smoothing | – | Metric `intersection_df_eps` |
| Kearns et al. 2018, *Fairness gerrymandering* | – | ✓ | ✓ rich subgroups | – | – | |
| Sagawa et al. 2020, *Group DRO* | – | ✓ | – | – | – | Non-private ref `groupdro` |
| Kim et al. 2025, *Doubly-regressing subgroup fairness* (arXiv 2510.21091) (verify) | – | ✓ | ✓ hierarchy | ✓ | – | Non-private hierarchy idea |

## 2. Consequence for our contribution (pivot per plan §2)

Private worst-group optimisation per se is **not novel** (2402.19437,
2602.10820, Noe et al.). We therefore position IG-DPSGD on what the screen did
not find:

1. **Intersectional cells under DP** – many tiny cells where noisy statistics
   are dominated by noise; *hierarchical shrinkage* of noisy cell estimates
   toward marginal parents derived by post-processing (zero extra budget).
2. **Exact joint accounting of gradient + group-statistics releases** as one
   subsampled Gaussian mechanism with an explicit *budget split* ρ
   (`src/fairdp/dp/accounting.py`) and an empirical study of optimal ρ.
3. **Evaluation contribution** – DP-induced disparate impact Δ_g at fixed ε on
   intersectional cells across 3 datasets, with per-group membership-inference
   leakage (does mitigation shift privacy risk onto minorities?) and an LLM
   comparison on the same tasks.

Mandatory: implement or faithfully reproduce 2602.10820 as a baseline once read;
tune every DP baseline's lr/clip separately (2510.01744).

## 3. LLM study references (to collect)
DiscrimEval (Tamkin et al. 2023); BBQ (Parrish et al. 2022); ConfAIde
(Mireshghallah et al. 2024); Min-K% / WikiMIA (Shi et al. 2024); PII leakage
(Lukas et al. 2023); DP fine-tuning of LMs (Yu et al. 2022).
