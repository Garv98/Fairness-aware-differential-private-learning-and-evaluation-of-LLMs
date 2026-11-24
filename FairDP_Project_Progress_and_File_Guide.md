# FairDP Research --- Project Progress & Experiment Documentation

**Project:** Fairness-Aware Differentially Private Learning and
Privacy--Fairness Evaluation of Modern LLMs\
**Current completed scope:** Credit-risk ML, DP-SGD experiments, utility
evaluation, and group fairness analysis.\
**LLM component:** Assigned to a teammate and not yet implemented in
this experimental sequence.

------------------------------------------------------------------------

## 1. Project objective and current scope

This project studies the interaction between model utility, differential
privacy, and group fairness in credit-risk classification. The current
implemented experiments use a credit-risk dataset and compare a
conventional baseline with differentially private MLP models trained at
multiple noise levels.

The current fairness audit uses the available proxy/group attributes
**age, income, and homeownership**. The dataset does not provide race or
gender attributes in the current experiment, so the results must not be
described as race- or gender-fairness results.

------------------------------------------------------------------------

## 2. Dataset

-   **Dataset:** Kaggle Credit Risk Dataset (laotse)
-   **Dataset page:**
    https://www.kaggle.com/datasets/laotse/credit-risk-dataset
-   **Original shape:** 32,581 rows × 12 columns
-   **Target:** `loan_status` (0 = non-default, 1 = default)
-   **Class distribution:** 25,473 non-default (78.18%); 7,108 default
    (21.82%)
-   **Split:** Stratified 80/20 train/test; `random_state=42`
-   **Processed feature dimensions:** 26 features after encoding
-   **Training set:** 25,932 samples
-   **Test set:** 6,484 samples

Preprocessing includes duplicate removal, numerical median imputation,
categorical most-frequent imputation, one-hot encoding, and standard
scaling. The preprocessing steps are not differentially private.
Therefore, the DP guarantee must be scoped to the DP training mechanism,
not claimed for the entire end-to-end data pipeline.

------------------------------------------------------------------------

## 3. Project structure and files

Root directory: `D:\\Ethical_AI\\FairDP_Research`

``` text
FairDP_Research/
├── data/
│   ├── credit_risk_dataset.csv
│   └── processed_data.csv
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_baseline_ml.ipynb
│   ├── 03_differential_privacy.ipynb
│   ├── 04_fairness_mitigation.ipynb
│   └── 05_final_comparison.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── dp_training.py
│   ├── fairness.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── baseline/
│   ├── dp/
│   └── fair_dp/
├── results/
│   ├── metrics.csv
│   ├── predictions.csv
│   └── graphs/
├── configs/
│   └── experiment_config.yaml
├── tests/
│   └── test_preprocessing.py
├── docs/
│   ├── methodology.md
│   └── references.md
├── requirements.txt
├── README.md
└── .gitignore
```

### File responsibilities

  -----------------------------------------------------------------------
  File / notebook                     Contents / role
  ----------------------------------- -----------------------------------
  `01_data_analysis.ipynb`            Dataset inspection, distributions,
                                      class balance, exploratory analysis

  `02_baseline_ml.ipynb`              Logistic Regression, standard MLP
                                      and weighted MLP baseline
                                      experiments and utility metrics

  `03_differential_privacy.ipynb`     DP training experiments,
                                      noise/clipping setup, privacy
                                      accountant outputs, and DP utility
                                      results

  `04_fairness_mitigation.ipynb`      Fairness subgroup evaluation,
                                      comparisons across
                                      age/income/homeownership, DP
                                      fairness results, tables and
                                      graph-generation cells

  `05_final_comparison.ipynb`         Intended final consolidation
                                      notebook; check its current
                                      contents before treating it as
                                      complete

  `src/preprocessing.py`              Shared preprocessing and train/test
                                      preparation functions, including
                                      `prepare_data()` and
                                      `prepare_fairness_data()`

  `src/models.py`                     Model definitions used by the
                                      experiments

  `src/train.py`                      Standard training routines

  `src/dp_training.py`                DP training logic and
                                      Opacus-related configuration

  `src/fairness.py`                   Group metric evaluation
                                      (`evaluate_group_metrics`)
                                      including accuracy, precision,
                                      recall, F1, selection rate, FPR and
                                      FNR

  `src/evaluation.py`                 Shared utility metric evaluation

  `src/visualization.py`              Plotting helpers, if used by the
                                      notebooks

  `configs/experiment_config.yaml`    Experiment parameters/configuration

  `tests/test_preprocessing.py`       Preprocessing checks/tests

  `docs/methodology.md`               Methodology documentation

  `docs/references.md`                Research references

  `results/`                          Exported metrics, subgroup CSVs,
                                      comparisons and plots

  `models/`                           Saved model checkpoints by
                                      experiment family
  -----------------------------------------------------------------------

**Note:** This is the project structure tracked in our working plan. The
existence and latest contents of every file have not been independently
inspected from the local computer in this chat.

------------------------------------------------------------------------

## 4. Implemented models and experiments

### Baseline models

1.  Logistic Regression
2.  Standard MLP
3.  Class-weighted MLP

The class-weighted MLP uses class weights to address class imbalance. It
is **not** group-aware fairness mitigation.

### Differentially private models

DP MLPs were trained with Opacus, gradient clipping, and Gaussian noise:

  -----------------------------------------------------------------------------------
  Configuration          Noise        Max     Epochs Batch size   Learning          δ
                    multiplier   gradient                             rate 
                                     norm                                  
  --------------- ------------ ---------- ---------- ---------- ---------- ----------
  DP-MLP 1.0               1.0        1.0         20         64      0.001       1e-5

  DP-MLP 1.5               1.5        1.0         20         64      0.001       1e-5

  DP-MLP 2.0               2.0        1.0         20         64      0.001       1e-5
  -----------------------------------------------------------------------------------

The reported privacy budget ε is obtained from the Opacus accountant
under the training configuration and its sampling/accounting
assumptions.

------------------------------------------------------------------------

## 5. Main utility results

Metrics are reported as proportions in the consolidated CSV (multiply by
100 for percentages).

  Model                   Accuracy   Precision   Recall       F1   ROC-AUC     ε
  --------------------- ---------- ----------- -------- -------- --------- -----
  Logistic Regression       0.8691      0.7728   0.5684   0.6550    0.8680   ---
  Standard MLP              0.9212      0.9530   0.6728   0.7888       --- 
  Weighted MLP              0.8896      0.7417   0.7595   0.7505       --- 
  DP-MLP Noise 1.0          0.8837      0.8138   0.6072   0.6955    1.3304 

DP noise-level results:

  Noise          ε   Accuracy   Precision   Recall       F1   ROC-AUC
  ------- -------- ---------- ----------- -------- -------- ---------
  1.0       1.3304     0.8837      0.8138   0.6072   0.6955    0.8852
  1.5       0.6640     0.8738      0.7863   0.5811   0.8784 
  2.0       0.4567     0.8760      0.8252   0.5494   0.8767 

------------------------------------------------------------------------

## 6. Fairness evaluation

Group metrics are calculated for age, income, and homeownership groups.
Metrics include subgroup sample count, accuracy, precision, recall, F1,
selection rate, FPR, and FNR.

**Important interpretation:** `Selection_Rate` in these tables is the
proportion predicted as default (class 1), not a loan-approval rate.

Homeownership `OTHER` contains only 18 test samples and should be
treated as highly unstable. For the main homeownership disparity
comparison, report the gap **excluding OTHER**, while optionally showing
the all-group gap as a sensitivity result.

### DP recall disparity comparison (percentage points)

Recall gap = maximum subgroup recall − minimum subgroup recall.

  Attribute                           Noise 1.0   Noise 1.5   Noise 2.0
  --------------------------------- ----------- ----------- -----------
  Age                                     12.93       11.07       11.81
  Income                                  31.78       40.38       38.04
  Homeownership (excluding OTHER)         47.57       43.83       44.17

Homeownership all-group gaps, including `OTHER`:

  Noise     All-group gap (pp)
  ------- --------------------
  1.0                    47.57
  1.5                    73.91
  2.0                    60.53

------------------------------------------------------------------------

## 7. Results CSV files created / tracked in the working record

The following names and results are documented in the experiment record.
Verify files on disk before submission or sharing.

### Main CSVs to use

  ------------------------------------------------------------------------------------------------------------
  CSV                                                          Purpose                 Recommended use
  ------------------------------------------------------------ ----------------------- -----------------------
  `results/model_performance_comparison.csv`                   Baseline model utility  Main
                                                               comparison              baseline-versus-model
                                                                                       table

  `results/model_fairness_comparison.csv`                      Baseline/weighted/DP    Main baseline fairness
                                                               Noise 1.0 fairness      comparison
                                                               comparison              

  `results/final_dp_privacy_utility_fairness_comparison.csv`   DP Noise 1.0/1.5/2.0    **Primary DP results
                                                               privacy, utility and    table**
                                                               fairness in one table   

  `results/dp_fairness_all_privacy_levels.csv`                 Age, income and         Fairness results table
                                                               homeownership disparity 
                                                               comparison across DP    
                                                               noise levels            

  `results/dp_privacy_level_utility_comparison.csv`            DP utility comparison   Supplementary; prefer
                                                               between noise levels    final consolidated
                                                               1.0 and 1.5 (original   table for all three
                                                               comparison)             levels

  `results/dp_fairness_noise_comparison.csv`                   DP fairness comparison  Supplementary;
                                                               for noise 1.0 and 1.5   superseded by
                                                                                       all-three-level
                                                                                       comparison

  `results/privacy_utility_tradeoff.csv`                       Standard MLP vs DP      Utility cost of DP
                                                               Noise 1.0 utility       
                                                               differences             

  `results/dp_noise_1_5_results.csv`                           Noise 1.5 utility       Supporting detail
                                                               evaluation results      

  `results/age_fairness_dp_noise_1_5.csv`                      Noise 1.5 age subgroup  Supporting subgroup
                                                               metrics                 detail

  `results/income_fairness_dp_noise_1_5.csv`                   Noise 1.5 income        Supporting subgroup
                                                               subgroup metrics        detail

  `results/homeownership_fairness_dp_noise_1_5.csv`            Noise 1.5 homeownership Supporting subgroup
                                                               subgroup metrics        detail

  `results/age_fairness_dp_noise_2_0.csv`                      Noise 2.0 age subgroup  Supporting subgroup
                                                               metrics                 detail

  `results/income_fairness_dp_noise_2_0.csv`                   Noise 2.0 income        Supporting subgroup
                                                               subgroup metrics        detail

  `results/homeownership_fairness_dp_noise_2_0.csv`            Noise 2.0 homeownership Supporting subgroup
                                                               subgroup metrics        detail
  ------------------------------------------------------------------------------------------------------------

Other files may exist (for example `metrics.csv`, `predictions.csv`,
per-model result files, and baseline subgroup CSVs). The exact current
contents and filesystem status need local verification.

### CSVs to use in the research paper

Use these as the core paper tables:

1.  **`model_performance_comparison.csv`** --- baseline and model
    utility.
2.  **`model_fairness_comparison.csv`** --- baseline/weighted/DP 1.0
    fairness.
3.  **`final_dp_privacy_utility_fairness_comparison.csv`** --- DP
    privacy--utility--fairness comparison across all three noise levels.
4.  **`dp_fairness_all_privacy_levels.csv`** --- detailed DP disparity
    values.

Use individual subgroup CSVs for verification, supplementary tables, or
appendix---not all in the main paper.

------------------------------------------------------------------------

## 8. Graphs: what to use in the paper

### Main-paper figures (recommended)

  ---------------------------------------------------------------------------------------------------
  Figure                  Intended content                       Filename/status
  ----------------------- -------------------------------------- ------------------------------------
  DP privacy comparison   ε vs noise multiplier                  `final_dp_privacy_comparison.png`
                                                                 --- image displayed; saving was not
                                                                 interrupted in the privacy plot
                                                                 section, but confirm file exists

  DP utility comparison   Accuracy/Precision/Recall/F1/ROC-AUC   `final_dp_utility_comparison.png`
                          vs noise                               --- graph displayed, save
                                                                 interrupted by `KeyboardInterrupt`;
                                                                 not confirmed saved

  DP fairness comparison  Age/income/homeownership recall gaps   `final_dp_fairness_comparison.png`
                          vs noise                               --- save status not confirmed from
                                                                 output

  Overall model utility   Logistic Regression, standard MLP,     Earlier graph generated; confirm
  comparison              weighted MLP, DP MLP                   filename on disk

  Privacy--utility        Utility loss from Standard MLP to DP   Earlier graph generated; confirm
  tradeoff                Noise 1.0                              filename on disk
  ---------------------------------------------------------------------------------------------------

### Supplementary figures

-   Age fairness comparisons across model types.
-   Income fairness comparisons across model types.
-   Homeownership fairness comparisons across model types.
-   DP Noise 1.0 vs 1.5 fairness graphs.
-   DP Noise 1.0 vs 1.5 utility graph.
-   Per-attribute DP noise-level fairness charts, if verified on disk.

**Graph-saving caution:** The utility figure's `savefig(..., dpi=300)`
was interrupted by a `KeyboardInterrupt` during PNG saving. The plot was
already displayed, so the data/plot was generated, but the file save was
not confirmed. A recovery cell was suggested, but its execution/output
was not confirmed. Recreate and save final graphs at 120--150 DPI if
needed, and close figures after saving.

------------------------------------------------------------------------

## 9. Suggested final files for paper

### Main Results section

-   `model_performance_comparison.csv`
-   `model_fairness_comparison.csv`
-   `final_dp_privacy_utility_fairness_comparison.csv`
-   `dp_fairness_all_privacy_levels.csv`

### Main figures

-   DP privacy comparison
-   DP utility comparison
-   DP fairness comparison
-   Overall model utility comparison
-   Standard vs DP utility tradeoff

### Appendix / supplementary material

-   Detailed age, income, homeownership subgroup CSVs
-   Per-noise DP utility result files
-   Additional fairness plots
-   Training/privacy-accounting outputs

------------------------------------------------------------------------

## 10. Completed work versus pending work

### Completed in this ML/DP workstream

-   Dataset loading and exploratory analysis
-   Preprocessing and train/test preparation
-   Logistic Regression baseline
-   Standard MLP
-   Class-weighted MLP
-   DP-MLP Noise 1.0, 1.5, 2.0 training
-   DP privacy accounting (ε, δ)
-   Model utility evaluation
-   Group fairness evaluation for age, income and homeownership
-   Consolidated DP comparison table
-   DP fairness disparity comparison table
-   Privacy/utility/fairness interpretation draft

### Pending / not claimed as completed

-   LLM fairness/privacy experiments (assigned to teammate)
-   Verify final graph files are saved
-   Verify all CSVs physically exist and contain expected data
-   Final paper integration and citation/reference checks
-   Final reproducibility review

------------------------------------------------------------------------

## 11. Reporting cautions

1.  Do not claim the entire preprocessing pipeline is differentially
    private; preprocessing is outside the DP mechanism used here.
2.  Do not claim race/gender fairness; those attributes are not
    available in the current dataset.
3.  Clarify that selection rate is predicted-default rate.
4.  Treat the `OTHER` homeownership group cautiously because it contains
    18 samples.
5.  Do not claim noise causes a monotonic fairness trend; disparity
    varies by attribute and noise configuration.
6.  These are single-run comparisons; repeat runs and confidence
    intervals would strengthen the study.
7.  For the DP noise-level fairness comparison, the primary
    homeownership gap excludes `OTHER`; report the all-group gap
    separately if needed.
8.  Standard MLP fairness values are from the fairness notebook's
    retrained model; do not silently assume they correspond exactly to
    the weights used in the earlier baseline utility table.
9.  The DP models were trained with the recorded Opacus configuration.
    State the accountant, sampling, δ, clipping, and noise details
    accurately in the methods section.

------------------------------------------------------------------------

## 12. Recommended research-paper organization for this workstream

-   Dataset and preprocessing
-   Baseline model performance
-   Differential privacy setup
-   DP privacy--utility comparison
-   Group fairness analysis
-   Joint privacy--utility--fairness discussion
-   Limitations
-   LLM experimental component (after teammate's results are ready)
