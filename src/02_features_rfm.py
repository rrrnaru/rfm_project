# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

IN_ALL = Path(r"D:\\rfm_project\\data\\interim\\events_all.csv")
IN_BUY = Path(r"D:\\rfm_project\\data\\interim\\buy_records.csv")
OUT   = Path(r"D:\\rfm_project\\data\\interim\\rfm_features.csv")

events = pd.read_csv(IN_ALL, parse_dates=["time"])  # type: ignore
buys   = pd.read_csv(IN_BUY,  parse_dates=["pay_time"])  # type: ignore

snapshot_date = events["time"].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# 全行为聚合
g_all = events.groupby("user_id").agg(
    pv_cnt=("behavior_type", lambda x: (x=="pv").sum()),
    cart_cnt=("behavior_type", lambda x: (x=="cart").sum()),
    fav_cnt=("behavior_type", lambda x: (x=="fav").sum()),
    buy_cnt=("behavior_type", lambda x: (x=="buy").sum()),
    active_days_all=("time",   lambda x: x.dt.date.nunique()),
    last_event=("time","max")
).reset_index()

# 购买聚合
if len(buys) == 0:
    rfm = g_all.copy()
    rfm["last_buy"] = pd.NaT
    rfm["F"] = 0
    rfm["M"] = 0
    rfm["unique_item_cnt"] = 0
    rfm["active_days_buy"] = 0
else:
    g_buy = buys.groupby("user_id").agg(
        last_buy=("pay_time","max"),
        F=("order_id_pseudo","nunique"),
        unique_item_cnt=("item_id","nunique"),
        active_days_buy=("pay_time", lambda x: x.dt.date.nunique())
    ).reset_index()
    rfm = g_all.merge(g_buy, on="user_id", how="left")

# R 定义（无购买用户用最后事件+1000 天兜底）
rfm["R"] = (snapshot_date - rfm["last_buy"]).dt.days
rfm["R"] = rfm["R"].fillna((snapshot_date - rfm["last_event"]).dt.days + 1000)
rfm["F"] = rfm["F"].fillna(0).astype(int)
rfm["M"] = rfm["unique_item_cnt"].fillna(0).astype(int)

# 转化率（除零保护）
rfm["pv2buy_rate"]   = rfm.apply(lambda r: r["buy_cnt"]/r["pv_cnt"]   if r["pv_cnt"]>0   else 0.0, axis=1)
rfm["cart2buy_rate"] = rfm.apply(lambda r: r["buy_cnt"]/r["cart_cnt"] if r["cart_cnt"]>0 else 0.0, axis=1)

cols = [
    "user_id","R","F","M",
    "pv_cnt","cart_cnt","fav_cnt","buy_cnt",
    "pv2buy_rate","cart2buy_rate",
    "active_days_all","active_days_buy"
]
rfm[cols].to_csv(OUT, index=False)
print(f"✅ RFM 特征完成 → {OUT}")