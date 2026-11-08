import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline

table = pd.read_csv("window_features.csv")
X = table[[c for c in table.columns if c.startswith(("delta_", "baseline_"))]]
y, groups = table["label"], table["participant"]


def loso_auc(labels):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          RandomForestClassifier(100, max_depth=4, random_state=42, class_weight="balanced", n_jobs=-1))
    p = cross_val_predict(model, X, labels, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    return roc_auc_score(labels, p)


observed = loso_auc(y)
rng = np.random.default_rng(0)
null = [loso_auc(y.groupby(groups).transform(lambda s: rng.permutation(s.to_numpy()))) for _ in range(100)]
print(f"observed AUC {observed:.3f}; null mean {np.mean(null):.3f}, 95th pct {np.percentile(null, 95):.3f}; "
      f"p = {(1 + sum(n >= observed for n in null)) / (1 + len(null)):.3f}")
