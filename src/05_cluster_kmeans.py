# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DF = r"D:\\rfm_project\\data\\interim\\rfm_labeled.csv"
df = pd.read_csv(DF)

X = df[["R","F","M"]].copy()
sc = StandardScaler()
Xz = sc.fit_transform(X)

km = KMeans(n_clusters=4, n_init=20, random_state=42)
df["rfm_cluster"] = km.fit_predict(Xz)

# 轮廓值/肘部法可在报告中讨论，这里聚类标签直接落盘

OUT = r"D:\\rfm_project\\data\\interim\\rfm_clustered.csv"
df.to_csv(OUT, index=False)
print(f"✅ 聚类完成 → {OUT}")