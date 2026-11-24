"""Train/val/test splitting and feature preprocessing.

Privacy scope note: the imputer/scaler/encoder are fitted on the training
split without DP. The formal (eps, delta) guarantee therefore covers the
training mechanism (DP-SGD + auxiliary releases) only, not preprocessing.
This mirrors common practice and is stated as a limitation in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .loaders import TabularDataset
from .. import groups as G


@dataclass
class Split:
    X: np.ndarray            # processed float32 features
    y: np.ndarray            # int labels
    groups: pd.DataFrame     # marginal group labels + "intersection" column
    raw: pd.DataFrame        # raw features (for LLM serialisation, audits)


@dataclass
class PreparedData:
    name: str
    train: Split
    val: Split
    test: Split
    feature_names: list[str]
    group_attrs: list[str]
    preprocessor: object = None      # fitted ColumnTransformer (for serving new inputs)
    binner: object = None            # fitted group binner (raw sensitive cols -> groups)


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))])
    return ColumnTransformer([("num", num, numerical), ("cat", cat, categorical)])


def prepare(ds: TabularDataset, seed: int = 42, test_size: float = 0.2,
            val_size: float = 0.1, group_attrs: list[str] | None = None) -> PreparedData:
    """Stratified (label x first group attribute) train/val/test split.

    Group binning thresholds (e.g. income quantiles) are learned on the
    training split only, then applied to val/test.
    """
    n = len(ds)
    idx = np.arange(n)
    spec = G.get_spec(ds.name)
    attrs = group_attrs or spec.default_attrs
    # Stratify on label x coarse primary group, computed with train-agnostic
    # fixed rules (quantile-free) to avoid leaking split information.
    strat_key = pd.Series(ds.y).astype(str) + "_" + spec.stratify_key(ds.sensitive).astype(str)
    # rare strata fall back to label-only stratification
    counts = strat_key.map(strat_key.value_counts())
    strat_key = strat_key.where(counts >= 10, pd.Series(ds.y).astype(str))

    idx_trval, idx_test = train_test_split(idx, test_size=test_size,
                                           random_state=seed,
                                           stratify=strat_key.to_numpy())
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(idx_trval, test_size=rel_val,
                                          random_state=seed,
                                          stratify=strat_key.to_numpy()[idx_trval])

    pre = build_preprocessor(ds.numerical, ds.categorical)
    X_tr = pre.fit_transform(ds.X.iloc[idx_train])
    binner = spec.fit(ds.sensitive.iloc[idx_train])

    def make(ix: np.ndarray, Xp: np.ndarray | None = None) -> Split:
        Xp = pre.transform(ds.X.iloc[ix]) if Xp is None else Xp
        grp = binner(ds.sensitive.iloc[ix]).reset_index(drop=True)
        grp = grp[attrs].copy()
        grp["intersection"] = G.intersect(grp, attrs)
        return Split(np.asarray(Xp, dtype=np.float32), np.asarray(ds.y)[ix].astype(int),
                     grp, ds.X.iloc[ix].reset_index(drop=True))

    return PreparedData(
        name=ds.name,
        train=make(idx_train, X_tr),
        val=make(idx_val),
        test=make(idx_test),
        feature_names=list(pre.get_feature_names_out()),
        group_attrs=list(attrs),
        preprocessor=pre,
        binner=binner,
    )
