# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

DF = r"D:\\rfm_project\\data\\interim\\rfm_labeled.csv"
df = pd.read_csv(DF)

feat = [
    "R","F","M",
    "pv_cnt","cart_cnt","fav_cnt","buy_cnt",
    "pv2buy_rate","cart2buy_rate",
    "active_days_all","active_days_buy"
]
X = df[feat]
X = X.fillna(0)
y = df["label"]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

sc = StandardScaler()
Xtr_s = sc.fit_transform(X_tr)
Xva_s = sc.transform(X_va)
Xte_s = sc.transform(X_te)

logit = LogisticRegression(max_iter=200, class_weight="balanced").fit(Xtr_s, y_tr)
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1
).fit(X_tr, y_tr)
xgb = XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc", random_state=42, n_jobs=-1
).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)


def p_at_k(y_true, y_score, k=0.3):
    n = int(len(y_score)*k); idx = np.argsort(-y_score)[:n]
    return y_true.iloc[idx].mean()

def lift_at_k(y_true, y_score, k=0.3):
    n = int(len(y_score)*k); idx = np.argsort(-y_score)[:n]
    return y_true.iloc[idx].mean() / y_true.mean()

res = []
for name, model, Xe in [
    ("logit", logit, Xte_s),
    ("rf", rf, X_te),
    ("xgb", xgb, X_te),
]:
    proba = model.predict_proba(Xe)[:,1]
    res.append((name,
                roc_auc_score(y_te, proba),
                p_at_k(y_te.reset_index(drop=True), pd.Series(proba), 0.3),
                lift_at_k(y_te.reset_index(drop=True), pd.Series(proba), 0.3)))

print("\n".join([f"{n}: AUC={a:.3f}, P@30%={p:.3f}, Lift@30%={l:.2f}" for n,a,p,l in res]))