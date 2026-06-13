import pandas as pd
import yaml
from pathlib import Path
import joblib
import useful_funcs

# ========== Opening config file ==========

path_yaml = Path("config.yaml")
try:

    with open(path_yaml, "r") as file:
        config = yaml.safe_load(file)

except FileNotFoundError:
    print("Config file not found")
    sys.exit(1)

# ========== creating dataframe, removing clients with fiber, creating path variables ==========

path_processed = Path(config["paths"]["processed"])
path_model_fiber = Path(config["paths"]["model_fiber"])
path_predictions = Path(config["paths"]["predictions"])
df = pd.read_csv(path_processed, index_col="customer_id")

df_nofiber = df[df["is_fiber"] == 0].drop(columns="is_fiber")

# ========== loading model and creating predictions ==========

model_fiber = joblib.load(path_model_fiber)

pred_prob = model_fiber.predict_proba(df_nofiber)[:, 1]

# ========== creating prediction dataframe and exporting as a csv ==========

df_prediction_fiber = pd.DataFrame({
    "Fiber": pred_prob
}, index=df_nofiber.index).sort_values(by="Fiber", ascending=False)

df_prediction_fiber.to_csv(path_predictions.joinpath("fiber.csv"), index="customer_id")

print(df_prediction_fiber)
