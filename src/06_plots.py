# -*- coding: utf-8 -*-
# 生成：reports/figures/ROC.png, Lift.png, Radar_RFM.png
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# 可选：如果你装了 xgboost 会一起画；没装自动跳过
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(r"D:\rfm_project")
DF_LABELED = ROOT / "data" / "interim" / "rfm_labeled.csv"
DF_CLUSTER = ROOT / "data" / "interim" / "rfm_clustered.csv"
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------- 读取数据 ----------
df = pd.read_csv(DF_LABELED)
feat = [
    "R","F","M",
    "pv_cnt","cart_cnt","fav_cnt","buy_cnt",
    "pv2buy_rate","cart2buy_rate",
    "active_days_all","active_days_buy"
]
X = df[feat].fillna(0)
y = df["label"]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

sc = StandardScaler()
Xtr_s = sc.fit_transform(X_tr)
Xva_s = sc.transform(X_va)
Xte_s = sc.transform(X_te)

# ---------- 训练模型（与 04 一致） ----------
models = {}
models["logit"] = LogisticRegression(max_iter=200, class_weight="balanced").fit(Xtr_s, y_tr)
models["rf"] = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1
).fit(X_tr, y_tr)
if HAS_XGB:
    models["xgb"] = XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc",
        random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

# ---------- ROC ----------
plt.figure()
for name, m in models.items():
    Xe = Xte_s if name == "logit" else X_te
    proba = m.predict_proba(Xe)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(FIGDIR / "ROC.png", dpi=180)
plt.close()

# ---------- Lift 曲线（优先用 xgb，没有就用 rf） ----------
def gains_lift(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = y_true.iloc[order].to_numpy()
    cum_pos = np.cumsum(y_sorted)
    perc = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    gains = cum_pos / y_true.sum()
    lift = gains / perc
    return perc, gains, lift

mdl = models.get("xgb", models["rf"])
Xe = X_te if mdl is models.get("rf") else X_te  # 两个都用 X_te
proba = mdl.predict_proba(Xe)[:, 1]
perc, gains, lift = gains_lift(y_te.reset_index(drop=True), pd.Series(proba))

plt.figure()
plt.plot(perc, lift)
plt.xlabel("Top portion of users")
plt.ylabel("Lift")
plt.title("Lift Curve")
plt.tight_layout()
plt.savefig(FIGDIR / "Lift.png", dpi=180)
plt.close()

# ---------- 聚类雷达图（R/F/M 三维画像） ----------
# 优先读 05 的聚类结果；如不存在就现场聚一遍
try:
    dcf = pd.read_csv(DF_CLUSTER)
except FileNotFoundError:
    dcf = df.copy()
    from sklearn.cluster import KMeans
    Xz = StandardScaler().fit_transform(dcf[["R","F","M"]])
    dcf["rfm_cluster"] = KMeans(n_clusters=4, n_init=20, random_state=42).fit_predict(Xz)

profile = dcf.groupby("rfm_cluster")[["R","F","M"]].mean()
# 归一化到 [0,1]，便于比较
prof = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

labels = ["R","F","M"]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

plt.figure()
for cid, row in prof.iterrows():
    vals = np.concatenate((row.values, [row.values[0]]))
    plt.polar(angles, vals, marker="o", label=f"cluster {cid}")
plt.thetagrids(angles[:-1] * 180/np.pi, labels)
plt.title("RFM Cluster Radar")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(FIGDIR / "Radar_RFM.png", dpi=180)
plt.close()

print(f"✅ 图已生成：{FIGDIR}")
