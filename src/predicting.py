import pandas as pd
import joblib
import yaml
import sys
import logging
from pathlib import Path

def prediction():
    """
    Use already trained model to predict possible clients,
    create best possible clients ranking data frame
    """ 

    try:

        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        logging.error("---------- config.yaml file not found ----------")
        sys.exit(1)


    path_model = Path(config["paths"]["model"])
    path_nofiber = Path(config["paths"]["nofiber"])
    path_fiber_prob = Path(config["paths"]["fiber_prob"])

    # ---------- Opening non fiber clients data frame ----------

    df = pd.read_csv(path_nofiber)

    # ---------- Loading model and creating "Migration Probability column" ----------

    logging.info("---------- Loading model and predicting ----------")

    try:
        model = joblib.load(path_model)

    except FileNotFoundError:
        logging.error("---------- You need to have a model to make the predictions ----------")
        logging.error("Try running preprocess.py -> train.py first in order to have the model")
        sys.exit(1)

    df['Migration Probability'] = model.predict_proba(df.drop(columns="CustomerID"))[:, 1]

    logging.info("---------- Prediction concluded ----------")

    # ---------- Creating target data frame ----------

    logging.info("---------- Creating data frame ----------")

    ranking_marketing = df[["CustomerID", "Migration Probability"]].sort_values(
        by="Migration Probability", 
        ascending=False
    )

    ranking_marketing.to_csv(path_fiber_prob , index=False)

    logging.info(f"---------- Prediction ranking created sucessfully at {path_fiber_prob} ----------")

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    prediction()
