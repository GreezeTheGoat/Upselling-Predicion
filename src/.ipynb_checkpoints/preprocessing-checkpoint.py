import pandas as pd
from pathlib import Path
import yaml
import sys

# ========== Opening config file ==========

path_yaml = Path("../src/config.yaml")
try:

    with open(path_yaml, "r") as file:
        config = yaml.safe_load(file)

except FileNotFoundError:
    print("Config file not found")
    sys.exit(1)

# ========== Creating path variables and opening dataset ==========

path_raw = Path(config["paths"]["raw"])
path_processed = Path(config["paths"]["processed"])
useless_columns = config["features"]["useless_columns"]
raw_standardization = config["data_standardization"]
extra_services_columns = config["features"]["extra_services"]

df = pd.read_csv(path_raw)

pd.set_option('future.no_silent_downcasting', True) # Boilerplate so i dont get stupid logs

# ========== Removing Churn clients and useless columns ==========


df = df[df["Churn Value"] == 0]

df = df.drop(columns=useless_columns)


# ========== Standardizing the data based on yaml file ==========

mapping = config["mapping"]
real_mapping = {}

for k, v in mapping.items():
    real_mapping[v] = k


df = df.rename(columns=real_mapping)

map_to_apply = {}
for std_key, raw_values in config["data_standardization"].items():
    if isinstance(raw_values, list):
        for val in raw_values:
            map_to_apply[val] = std_key
    else:
        map_to_apply[raw_values] = std_key

df = df.replace(map_to_apply)


# ========== Creating usefull columns and formating the entrys to lower case ==========

df["is_fiber"] = (df["internet_service"] == "fiber_optic").astype(int)
df = df.drop(columns=["internet_service"])

for col in config["features"]["nominal_categoricals"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

df["Total Extra Services"] = df[extra_services_columns].eq("yes").sum(axis = 1)


# ========== Dumping the dataframe into a csv file ==========

df.set_index("customer_id", inplace=True)

df.to_csv(path_processed, index=True)
