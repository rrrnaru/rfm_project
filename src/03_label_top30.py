# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler

rfm = pd.read_csv(r"D:\\rfm_project\\data\\interim\\rfm_features.csv")

sc = StandardScaler()
X = sc.fit_transform(rfm[["R","F","M"]])
rfm["zR"], rfm["zF"], rfm["zM"] = -X[:,0], X[:,1], X[:,2]  # R 取负
rfm["score"] = 0.3*rfm["zR"] + 0.3*rfm["zF"] + 0.4*rfm["zM"]

cut = rfm["score"].quantile(0.70)
rfm["label"] = (rfm["score"] >= cut).astype(int)

rfm.to_csv(r"D:\\rfm_project\\data\\interim\\rfm_labeled.csv", index=False)
print("✅ 打标完成 → data/interim/rfm_labeled.csv")