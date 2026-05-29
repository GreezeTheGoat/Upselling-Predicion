import pandas as pd
import joblib

def prediction():

    # ---------- Opening non fiber clients data frame ----------

    path_processed = "../data/processed/"

    df = pd.read_csv(f"{path_processed}nf_clients.csv")

    # ---------- Loading model and creating "Migration Probability column" ----------

    print("---------- Loading model and predicting ----------")

    model = joblib.load('../models/random_forest_final.pkl')

    df['Migration Probability'] = model.predict_proba(df.drop(columns="CustomerID"))[:, 1]

    print("---------- Prediction concluded ----------")

    # ---------- Creating target data frame ----------

    print("---------- Creating data frame ----------")

    ranking_marketing = df[["CustomerID", "Migration Probability"]].sort_values(
        by="Migration Probability", 
        ascending=False
    )

    ranking_marketing.to_csv("../data/processed/ranking_fiber_prob.csv", index=False)

    print("---------- Prediction ranking created sucessfully at ../data/processed/ranking_fiber_prob.csv ----------")

if __name__ == "__main__":
    prediction()
