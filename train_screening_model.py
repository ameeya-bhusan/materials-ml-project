import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from lightgbm import LGBMRegressor

df = pd.read_parquet("data/processed/li_electrodes_clean.parquet")
df = StrToComposition().featurize_dataframe(df, "framework_formula")
df = ElementProperty.from_preset("magpie").featurize_dataframe(df, "composition")
feat_cols = [c for c in df.columns if c.startswith("MagpieData")]

model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(df[feat_cols].values, df["average_voltage"].values)

# bundle everything the app needs
bundle = {
    "model": model,
    "feat_cols": feat_cols,
    "conformal_halfwidth": 0.899,   # the 90% interval from your conformal step
    "coverage": 0.90,
}
joblib.dump(bundle, "models/screening_model.joblib")
print("saved models/screening_model.joblib")