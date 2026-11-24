# Data

Datasets are **not** committed (`.gitignore` excludes `data/*.csv`).

| Dataset | How to obtain | Path |
|---|---|---|
| Credit Risk (Kaggle, laotse) | Download from https://www.kaggle.com/datasets/laotse/credit-risk-dataset | `data/credit_risk_dataset.csv` |
| UCI Adult | Downloaded automatically from OpenML on first use | `data/openml/` |
| ACS Income / Employment (folktables) | Downloaded automatically from the US Census on first use | `data/acs/` |

Cleaning applied to the credit data (`fairdp.data.loaders.load_credit`):
* exact duplicate rows are dropped
* rows with `person_age > 100` are dropped
* `person_emp_length > age − 14` is set to NaN, then median-imputed
