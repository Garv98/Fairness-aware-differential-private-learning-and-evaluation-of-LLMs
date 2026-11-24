# FairDP Research --- Results and Interpretation

This document summarizes the **observed results** from the credit-risk
ML and differential-privacy experiments and provides a cautious
interpretation suitable for drafting the results/discussion section. The
LLM experiments are not included because they are being completed
separately.

------------------------------------------------------------------------

## 1. Dataset and evaluation context

The experiment uses the Kaggle Credit Risk Dataset (laotse), with
`loan_status` as the target: 0 = non-default and 1 = default. The
dataset has 32,581 original rows and 12 columns. The observed target
distribution is 25,473 non-default records (78.18%) and 7,108 default
records (21.82%).

The prepared data contains 26 model features. A stratified 80/20 split
with random state 42 produced 25,932 training examples and 6,484 test
examples. Group fairness is evaluated by age, income, and homeownership.

------------------------------------------------------------------------

## 2. Baseline model utility results

  Model                   Accuracy   Precision   Recall       F1   ROC-AUC
  --------------------- ---------- ----------- -------- -------- ---------
  Logistic Regression       86.91%      77.28%   56.84%   65.50%    86.80%
  Standard MLP              92.12%      95.30%   67.28%   78.88%    91.65%
  Weighted MLP              88.96%      74.17%   75.95%   75.05%    91.80%

### Interpretation

-   Standard MLP achieved the highest accuracy, precision, and F1 among
    these baseline models.
-   Weighted MLP produced the highest recall and ROC-AUC in this
    recorded comparison.
-   The weighted model's recall improvement comes with lower precision
    and accuracy than Standard MLP.
-   Class weighting addresses class imbalance; it should not be
    described as group-fairness mitigation.

The target is imbalanced, so accuracy alone can hide missed defaults.
Precision and recall are especially useful for understanding false
alarms versus missed default cases.

------------------------------------------------------------------------

## 3. Differential privacy: privacy and utility

The three DP MLPs use noise multipliers 1.0, 1.5, and 2.0, with maximum
gradient norm 1.0, 20 epochs, batch size 64, learning rate 0.001, and δ
= 1e-5.

  Noise multiplier          ε   Accuracy   Precision   Recall       F1   ROC-AUC
  ------------------ -------- ---------- ----------- -------- -------- ---------
  1.0                  1.3304     88.37%      81.38%   60.72%   69.55%    88.52%
  1.5                  0.6640     87.38%      78.63%   58.11%   66.83%    87.84%
  2.0                  0.4567     87.60%      82.52%   54.94%   65.96%    87.67%

### Interpretation

-   Increasing the noise multiplier reduced the recorded ε: 1.3304 →
    0.6640 → 0.4567. For the same δ and accounting assumptions, the
    lower ε corresponds to a stronger privacy guarantee.
-   Accuracy remained within 87.38--88.37%, a relatively narrow range
    across these DP configurations.
-   Recall and F1 declined across the tested configurations. The Noise
    2.0 model has the lowest recall and F1 in this set.
-   Precision was non-monotonic: Noise 2.0 had the highest DP precision
    (82.52%), while Noise 1.5 had the lowest (78.63%).
-   ROC-AUC remained close (87.67--88.52%) across the DP models.

This is a privacy--utility trade-off in the observed experiments, not
proof that every metric must decline monotonically as noise increases.

------------------------------------------------------------------------

## 4. Privacy comparison with the non-DP Standard MLP

Recorded Standard MLP metrics: Accuracy 92.12%, Precision 95.30%, Recall
67.28%, F1 78.88%, ROC-AUC 91.65%.

Recorded DP Noise 1.0 metrics: Accuracy 88.37%, Precision 81.38%, Recall
60.72%, F1 69.55%, ROC-AUC 88.52%.

  Metric        Standard MLP   DP Noise 1.0   Difference (DP − Standard)
  ----------- -------------- -------------- ----------------------------
  Accuracy            92.12%         88.37%                     −3.75 pp
  Precision           95.30%         81.38%                    −13.92 pp
  Recall              67.28%         60.72%                     −6.56 pp
  F1                  78.88%         69.55%                     −9.33 pp
  ROC-AUC             91.65%         88.52%                     −3.13 pp

The DP Noise 1.0 model showed lower utility than the recorded Standard
MLP on each listed metric. This comparison is descriptive; model run
differences and the exact training/evaluation setup should be disclosed.

------------------------------------------------------------------------

## 5. Fairness results: Standard MLP

### Age groups

  Group      Samples   Recall      FNR
  -------- --------- -------- --------
  Young        3,013   69.54%   30.46%
  Middle       2,744   70.03%   29.97%
  Older          727   68.12%   31.88%

Recorded age recall gap: **1.91 percentage points**.

### Income groups

  Group      Samples   Recall      FNR
  -------- --------- -------- --------
  Low          1,577   77.22%   22.78%
  Middle       3,189   65.01%   34.99%
  High         1,718   56.49%   43.51%

Recorded income recall gap: **20.73 pp**. The High-income subgroup had
the lowest recall in this Standard MLP evaluation.

### Homeownership groups

  Group        Samples   Recall      FNR
  ---------- --------- -------- --------
  MORTGAGE       2,681   40.35%   59.65%
  OTHER             18   83.33%   16.67%
  OWN              521   69.57%   30.43%
  RENT           3,264   79.30%   20.70%

Mortgage-versus-Rent recall gap: **38.95 pp**. The `OTHER` group has
only 18 samples and is not a stable basis for strong conclusions.

### Interpretation

Age recall was relatively similar among the three age groups in the
Standard MLP evaluation. Income and homeownership had larger recall
differences. These are observed group-level performance disparities;
they do not by themselves establish the cause of the disparity or prove
discriminatory behavior.

------------------------------------------------------------------------

## 6. Fairness results: class-weighted MLP

Recorded results:

-   Age recall gap: 3.10 pp (Standard MLP: 1.91 pp)
-   Income recall gap: 23.36 pp (Standard MLP: 20.73 pp)
-   Income selection-rate gap: 36.69 pp
-   Homeownership mortgage--rent recall gap: 32.04 pp (Standard MLP:
    38.95 pp)

### Interpretation

Class weighting improved the mortgage-versus-rent recall-gap measure in
this recorded comparison, but it increased age and income recall gaps
relative to the Standard MLP. This illustrates that changing class
weights is not a universal fairness intervention. Because weights are
based on class labels rather than protected-group membership, this model
is not group-aware fairness mitigation.

------------------------------------------------------------------------

## 7. DP fairness comparison across privacy configurations

Recall disparity is calculated as the maximum subgroup recall minus the
minimum subgroup recall, expressed in percentage points.

### Main comparison (homeownership excludes OTHER)

  Attribute         Noise 1.0   Noise 1.5   Noise 2.0
  --------------- ----------- ----------- -----------
  Age                12.93 pp    11.07 pp    11.81 pp
  Income             31.78 pp    40.38 pp    38.04 pp
  Homeownership      47.57 pp    43.83 pp    44.17 pp

### Homeownership, including OTHER

  Noise     All-group disparity
  ------- ---------------------
  1.0                  47.57 pp
  1.5                  73.91 pp
  2.0                  60.53 pp

### Interpretation

-   Age disparities were smaller than income and homeownership
    disparities across the DP configurations in this recorded
    comparison.
-   Income recall disparity was largest at Noise 1.5 (40.38 pp), not at
    the highest noise multiplier.
-   Homeownership disparity excluding OTHER was lowest at Noise 1.5
    (43.83 pp) and changed only modestly between 1.5 and 2.0.
-   Including the small `OTHER` subgroup makes the all-group
    homeownership disparity much larger at Noise 1.5 and 2.0. This
    sensitivity is driven in part by the very small subgroup size and
    should be disclosed.
-   The fairness outcomes do not follow one monotonic trend with
    increasing noise. More noise did not uniformly increase every
    fairness disparity.

------------------------------------------------------------------------

## 8. Joint privacy--utility--fairness interpretation

No tested DP configuration is best on every metric.

-   **Noise 1.0:** Highest recall and F1 among DP models, and lowest
    income disparity among the three; but it has the highest ε and the
    largest homeownership gap (excluding OTHER).
-   **Noise 1.5:** Lowest age disparity and lowest homeownership
    disparity (excluding OTHER); but highest income disparity and lowest
    accuracy/precision in the DP comparison.
-   **Noise 2.0:** Lowest ε (strongest privacy of the three
    configurations under the recorded accounting), and highest precision
    among DP models; however, it has the lowest recall and F1.

The results motivate selecting a DP configuration based on the
application's priorities rather than one headline metric. A credit-risk
system must consider the cost of missed defaults, false positive
decisions, privacy requirements, and subgroup disparities jointly.

------------------------------------------------------------------------

## 9. Main research observations

1.  **Privacy budget:** Increasing the tested noise multiplier reduced
    the recorded ε.
2.  **Utility:** Accuracy and ROC-AUC stayed relatively stable, while
    recall and F1 were lower at higher noise levels in the recorded
    runs.
3.  **Fairness:** Group recall disparities varied by attribute and noise
    level; there was no uniform fairness trend.
4.  **Attribute sensitivity:** Income and homeownership disparities were
    larger than age disparities in these experiments.
5.  **Class weighting:** Weighted training shifted the utility/fairness
    profile but did not improve every group gap.
6.  **Joint assessment:** Privacy, utility, and fairness should be
    evaluated together; a privacy improvement alone does not guarantee
    improved fairness.

------------------------------------------------------------------------

## 10. Limitations and caveats

-   These are single-run results per model/configuration. Random seeds,
    repeated runs, uncertainty intervals, and significance tests would
    improve robustness.
-   The data contains age, income, and homeownership for this fairness
    analysis, but no race or gender variables; claims must remain
    limited to the evaluated groups.
-   The `OTHER` homeownership subgroup has only 18 test examples, so
    metrics are unstable.
-   The preprocessing pipeline was not DP. Do not describe the whole
    pipeline as end-to-end differentially private.
-   The experiments use one dataset; generalization to other datasets or
    real-world lending settings is not established.
-   The DP accountant's ε depends on the implemented sampling,
    accountant, and training assumptions. Report the exact accountant
    and configuration from the notebook.
-   Fairness gaps are descriptive metrics and do not establish causal
    discrimination.
-   Standard MLP fairness values were measured from a retrained
    fairness-notebook model; do not assume its exact weights are
    identical to the model used for baseline utility.
-   Selection rate is the predicted-default rate (positive class), not
    loan approval rate.
-   The LLM fairness/privacy evaluation is a separate workstream and is
    not covered by these findings.

------------------------------------------------------------------------

## 11. Draft results-section paragraph

Across the tested DP-MLP configurations, increasing the noise multiplier
reduced the reported privacy budget from ε = 1.3304 to ε = 0.4567 at δ =
10⁻⁵. Accuracy remained in the range of 87.38--88.37%, while recall
decreased from 60.72% to 54.94% and F1-score from 69.55% to 65.96%
across the Noise 1.0 to Noise 2.0 configurations. Group fairness
analysis revealed substantial variation in recall disparities across
age, income, and homeownership. Age disparity was comparatively smaller,
while income and homeownership disparities were larger. Fairness did not
change monotonically with noise, indicating that stronger privacy in
these experiments did not consistently produce either better or worse
group fairness. The findings support joint evaluation of privacy,
predictive utility, and group performance rather than reliance on any
single metric.

------------------------------------------------------------------------

## 12. Suggested paper tables and figures

### Main tables

-   Baseline model utility comparison
-   Baseline/weighted/DP Noise 1.0 fairness comparison
-   DP privacy--utility--fairness comparison across Noise 1.0, 1.5, and
    2.0
-   DP subgroup recall-disparity table

### Main figures

-   Privacy budget versus noise multiplier
-   Utility metrics versus noise multiplier
-   Age, income, and homeownership recall disparities versus noise
    multiplier
-   Overall baseline-model utility comparison
-   Standard MLP versus DP Noise 1.0 privacy--utility trade-off

### Supplementary material

-   Detailed subgroup metric CSVs for each DP noise configuration
-   Additional per-attribute fairness charts
-   Per-model training and privacy-accounting outputs

Verify graph filenames and files exist before inserting them into the
paper. In particular, the utility graph was displayed but its 300-DPI
save was interrupted, and the recovery save was not confirmed.
