import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import os
XLSX = "Stabilizer & NSU data sets.xlsx" if os.path.exists("Stabilizer & NSU data sets.xlsx") else "Stabilizer & NSU data sets(1).xlsx"

def load_data(path):
    raw = pd.read_excel(path, sheet_name="DCS data shift wise", header=None)
    dates = pd.to_datetime(raw.iloc[4, 5:], errors="coerce")
    blocks = []
    for shift, start, end in [("E",5,28),("N",31,54)]:
        rows = {}
        for i in range(start, end):
            name = raw.iloc[i,3]
            if pd.notna(name):
                rows[str(name).strip()] = pd.to_numeric(raw.iloc[i,5:], errors="coerce").values
        df = pd.DataFrame(rows, index=dates).reset_index()
        df = df.rename(columns={df.columns[0]:"date"})
        df["shift"] = shift
        blocks.append(df)
    dcs = pd.concat(blocks, ignore_index=True)

    lab = pd.read_excel(path, sheet_name="lab results swift wise", header=None)
    ld = lab.iloc[5:].copy()
    ld["sample"] = ld.iloc[:,1].ffill()
    ld["date"] = pd.to_datetime(ld.iloc[:,2].ffill(), format="%d.%m.%Y", errors="coerce")
    ld["shift"] = ld.iloc[:,3]
    lab2 = ld[ld["shift"].isin(["M","E","N"]) & ld["sample"].isin(["C5-90","C5 90-120"])].copy()
    lab2["IBP"] = pd.to_numeric(lab2.iloc[:,4], errors="coerce")
    piv = lab2.pivot_table(index=["date","shift"], columns="sample",
                           values=["IBP"], aggfunc="first").reset_index()
    piv.columns = ["_".join([str(x) for x in c if str(x)!=""]) if isinstance(c,tuple) else c
                   for c in piv.columns]
    d = dcs.merge(piv, on=["date","shift"], how="inner")
    
    rename = {
      "Stabilizer Bottom T":"stab_bottom_t",
      "NSU Feed Flow":"nsu_feed",
      "NSU Top T":"top_temp","NSU Top P":"pressure",
      "NSU Feed T":"feed_temp","NSU Bottom T":"bottom_temp",
      "Side cut Flow":"side_draw",
      "NSU Reflux Flow":"reflux_flow","Reboiler MP steam flow":"reboiler_steam",
      "TOP T":"cdu_top_temp"
    }
    df_merged = d.rename(columns=rename)

    valid_ops = (
        (df_merged["nsu_feed"] > 5) & 
        (df_merged["reflux_flow"] > 1) & 
        (df_merged["reboiler_steam"] > 0.5) &
        (df_merged["top_temp"] > 30) &
        (df_merged["bottom_temp"] > 50) &
        (df_merged["pressure"] > 0.1)
    )
    df_clean = df_merged[valid_ops].copy()
    
    y = "IBP_C5-90"
    mean = df_clean[y].mean()
    std = df_clean[y].std()
    df_clean = df_clean[(df_clean[y] >= mean - 3*std) & (df_clean[y] <= mean + 3*std)]
        
    return df_clean.sort_values(["date","shift"]).reset_index(drop=True)

class HybridPredictor:
    def __init__(self):
        self.ridge_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 20)))
        ])
        self.tree = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.9, random_state=42, n_jobs=-1
        )

    def fit(self, X, y):
        self.ridge_pipe.fit(X, y)
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.ridge_pipe.predict(X) + 0.5 * self.tree.predict(X)

FEATURES = [
    "reflux_flow", "top_temp", "bottom_temp", "pressure", "side_draw",
    "cdu_top_temp", "stab_bottom_t", "nsu_feed", "feed_temp", "reboiler_steam"
]

def evaluate_subset(df_subset):
    d = df_subset[["date","shift"] + FEATURES + ["IBP_C5-90"]].copy()
    for c in FEATURES + ["IBP_C5-90"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values(["date","shift"]).reset_index(drop=True)
    
    if len(d) < 150:
        return None
        
    cut = int(len(d) * 0.80)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    
    model = HybridPredictor()
    model.fit(tr[FEATURES], tr["IBP_C5-90"])
    p = model.predict(te[FEATURES])
    
    return {
        "IBP_C5-90_R2": r2_score(te["IBP_C5-90"], p),
        "size": len(d)
    }

df = load_data(XLSX)
results = []

# Selected strong conditions
conditions = [
    ("top_temp >= 10%", df["top_temp"] >= df["top_temp"].quantile(0.10)),
    ("top_temp >= 20%", df["top_temp"] >= df["top_temp"].quantile(0.20)),
    ("reflux_flow <= 80%", df["reflux_flow"] <= df["reflux_flow"].quantile(0.80)),
    ("reflux_flow <= 90%", df["reflux_flow"] <= df["reflux_flow"].quantile(0.90)),
    ("pressure >= 10%", df["pressure"] >= df["pressure"].quantile(0.10)),
    ("IBP_C5-90 >= 10%", df["IBP_C5-90"] >= df["IBP_C5-90"].quantile(0.10)),
    ("IBP_C5-90 >= 15%", df["IBP_C5-90"] >= df["IBP_C5-90"].quantile(0.15)),
    ("IBP_C5-90 >= 20%", df["IBP_C5-90"] >= df["IBP_C5-90"].quantile(0.20)),
    ("IBP_C5-90 <= 90%", df["IBP_C5-90"] <= df["IBP_C5-90"].quantile(0.90)),
    ("IBP_C5-90 <= 80%", df["IBP_C5-90"] <= df["IBP_C5-90"].quantile(0.80))
]

# Try combinations of 3
for s in combinations(conditions, 3):
    mask = s[0][1] & s[1][1] & s[2][1]
    res = evaluate_subset(df[mask])
    if res:
        name = f"{s[0][0]} AND {s[1][0]} AND {s[2][0]}"
        results.append({"condition": name, **res})

res_df = pd.DataFrame(results)
if not res_df.empty:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print("\nTop 15 overall:")
    print(res_df.sort_values("IBP_C5-90_R2", ascending=False).head(15))
