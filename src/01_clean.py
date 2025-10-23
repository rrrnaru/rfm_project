# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

RAW = r"D:\rfm_project\data\raw\transactions.csv"
OUT_DIR = Path(r"D:\rfm_project\data\interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = ["user_id","item_id","category_id","behavior_type","timestamp"]

# --- 1) 读数据（兼容：有表头/无表头） ---
try:
    # 情况A：CSV 有表头且列名匹配
    df = pd.read_csv(RAW, usecols=EXPECTED)
    from_header = True
except Exception as e:
    # 情况B：CSV 无表头（UserBehavior 原始文件常见），按我们给定列名读
    df = pd.read_csv(
        RAW,
        header=None,
        names=EXPECTED,
        usecols=range(len(EXPECTED))  # 0..4 列
    )
    from_header = False

# --- 2) 基本清洗 ---
df = df.dropna(subset=["user_id","item_id","behavior_type","timestamp"]).copy()
df["behavior_type"] = df["behavior_type"].astype(str).str.lower()
df = df[df["behavior_type"].isin(["pv","cart","fav","buy"])].copy()

# --- 3) 时间戳 -> datetime（秒级 Unix） ---
# （若你的样本是毫秒级，可改 unit="ms"）
df["time"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
df = df.dropna(subset=["time"]).copy()

# --- 4) 伪订单ID（对 buy 去重） ---
df_buy = df[df["behavior_type"] == "buy"].copy()
df_buy["order_id_pseudo"] = (
    df_buy["user_id"].astype(str) + "_" +
    df_buy["item_id"].astype(str) + "_" +
    df_buy["time"].dt.floor("s").astype(str)
)
df_buy = df_buy.sort_values("time").drop_duplicates("order_id_pseudo", keep="first")

# --- 5) 导出 ---
(df
 .to_csv(OUT_DIR / "events_all.csv", index=False))

(df_buy
 .rename(columns={"time":"pay_time"})
 [["user_id","item_id","category_id","pay_time","order_id_pseudo"]]
 .to_csv(OUT_DIR / "buy_records.csv", index=False))

print("✅ 清洗完成：")
print(f"表头检测：{'使用原表头' if from_header else '无表头，已按约定列名载入'}")
print(f"全行为：{len(df):,} 行 → {OUT_DIR/'events_all.csv'}")
print(f"购买明细：{len(df_buy):,} 行 → {OUT_DIR/'buy_records.csv'}")