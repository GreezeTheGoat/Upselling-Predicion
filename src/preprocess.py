import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
import logging
from pathlib import Path

def preprocess():

    """
    Loads raw Telco data, cleans columns, maps categorical variables to numeric,
    creates engineered features, and splits data into train/test sets.
    """

    try:

        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        logging.error("---------- config.yaml file not found ----------")
        sys.exit(1)

    path_raw = Path(config["paths"]["raw"])
    path_processed = Path(config["paths"]["processed"])
    path_nofiber = Path(config["paths"]["nofiber"])
    nofiber_columns = config["features"]["nofiber_columns"]
    extra_services_columns = config["features"]["extra_services"]
    useless_columns = config["features"]["useless_columns"]


    # ---------- Opening Raw data and removing useless columns ----------

    df = pd.read_csv(path_raw)

    df = df[df["Churn Value"] == 0]

    pd.set_option('future.no_silent_downcasting', True)

    df = df.drop(columns = useless_columns)


    # ---------- Replacing strings with numbers ----------

    numeric_map = {
        "Yes": 1,
        "Male": 1,
        "No": 0,
        "No internet service": 0,
        "No phone service": 0,
        "Female": 0,
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 0,
        "Electronic check": 1,
        "Mailed check": 2,
        "DSL": 1,
        "Fiber optic": 2,
    }

    df = df.replace(numeric_map)

    # ---------- Creating "Total Extra Services" and "is_fiber" columns ----------

    df["Total Extra Services"] = df[extra_services_columns].sum(axis = 1)


    df["is_fiber"] = (df["Internet Service"] == 2).astype(int) # Removing ghost clients

    # ---------- Creating training and testing processed data

    X = df.drop(columns = ["is_fiber", "Internet Service", "CustomerID"])
    y = df["is_fiber"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    X_train.to_csv(path_processed.joinpath("X_train.csv"), index=False)
    X_test.to_csv(path_processed.joinpath("X_test.csv"), index=False)
    y_train.to_csv(path_processed.joinpath("y_train.csv"), index=False)
    y_test.to_csv(path_processed.joinpath("y_test.csv"), index=False)

    # ---------- Creating no fiber clients dataset ----------

    df_nofiber = df[df["is_fiber"] == 0].copy()
    df_nofiber = df_nofiber.drop(columns = ["is_fiber", "Internet Service"])


    df_nofiber = df_nofiber[nofiber_columns]

    df_nofiber.to_csv(path_nofiber, index=False)

    logging.info("---------- Processed raw data with sucsess ----------")

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    preprocess()

