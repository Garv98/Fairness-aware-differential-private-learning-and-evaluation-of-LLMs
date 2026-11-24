import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_group_metrics(y_true, y_pred, groups):
    results = []

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    data = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": groups
    })

    for group_name, group_data in data.groupby("group", observed=True):

        actual = group_data["y_true"]
        predicted = group_data["y_pred"]

        # Confusion matrix components
        tp = ((actual == 1) & (predicted == 1)).sum()
        fp = ((actual == 0) & (predicted == 1)).sum()
        tn = ((actual == 0) & (predicted == 0)).sum()
        fn = ((actual == 1) & (predicted == 0)).sum()

        results.append({
            "Group": group_name,
            "Samples": len(group_data),
            "Accuracy": accuracy_score(actual, predicted),
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "F1": f1_score(
                actual, predicted, zero_division=0
            ),
            "Selection_Rate": predicted.mean(),
            "FPR": fp / (fp + tn) if (fp + tn) else 0,
            "FNR": fn / (fn + tp) if (fn + tp) else 0
        })

    return pd.DataFrame(results)