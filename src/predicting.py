import pandas as pd
import joblib
import yaml
import sys

def prediction():

    try:

        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        print("---------- config.yaml file not found ----------")
        sys.exit(1)


    path_processed = config["paths"]["processed"]
    path_model = config["paths"]["model"]
    path_fiber_prob = config["paths"]["fiber_prob"]

    # ---------- Opening non fiber clients data frame ----------

    df = pd.read_csv(path_processed + "nf_clients.csv")

    # ---------- Loading model and creating "Migration Probability column" ----------

    print("---------- Loading model and predicting ----------")

    try:
        model = joblib.load(path_model)

    except FileNotFoundError:
        print("---------- You need to have a model to make the predictions ----------")
        print("Try running preprocess.py -> train.py first in order to have the model")
        sys.exit(1)

    df['Migration Probability'] = model.predict_proba(df.drop(columns="CustomerID"))[:, 1]

    print("---------- Prediction concluded ----------")

    # ---------- Creating target data frame ----------

    print("---------- Creating data frame ----------")

    ranking_marketing = df[["CustomerID", "Migration Probability"]].sort_values(
        by="Migration Probability", 
        ascending=False
    )

    ranking_marketing.to_csv(path_fiber_prob , index=False)

    print(f"---------- Prediction ranking created sucessfully at "+ path_fiber_prob + " ----------")

if __name__ == "__main__":
    prediction()
