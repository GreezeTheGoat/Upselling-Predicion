import pandas as pd
import yaml
from pathlib import Path
import joblib

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

path_model_fiber = Path(config["paths"]["models"]).joinpath("fiber.pkl")
path_model_str_tv = Path(config["paths"]["models"]).joinpath("str_tv.pkl")
path_model_str_movies = Path(config["paths"]["models"]).joinpath("str_movies.pkl")

path_predictions = Path(config["paths"]["predictions"])


df = pd.read_csv(path_processed, index_col="customer_id")


# ========== Creating predicting functions ==========

def predict_fiber(df, model_path, save_csv=False):
    df_nofiber = df[df["is_fiber"] == 0].drop(columns="is_fiber")
    
    # ========== loading model and creating predictions ==========
    
    model_fiber = joblib.load(model_path)
    
    pred_prob = model_fiber.predict_proba(df_nofiber)[:, 1]

    # ========== creating prediction dataframe and exporting as a csv ==========
    
    df_prediction_fiber = pd.DataFrame({
        "Fiber": pred_prob
    }, index=df_nofiber.index).sort_values(by="Fiber", ascending=False)

    if save_csv:
        df_prediction_fiber.to_csv(path_predictions.joinpath("fiber.csv"), index="customer_id")
    
    return df_prediction_fiber


def predict_str_tv(df, model_path, save_csv=False):
    df_no_str_tv = df[df["streaming_tv"] == 0].drop(columns="streaming_tv")

    # ========== loading model and creating predictions ==========
    
    model_str_tv = joblib.load(model_path)
    
    pred_prob = model_str_tv.predict_proba(df_no_str_tv)[:, 1]

    # ========== creating prediction dataframe and exporting as a csv ==========
    
    df_prediction_str_tv = pd.DataFrame({
        "Streaming TV": pred_prob
    }, index=df_no_str_tv.index).sort_values(by="Streaming TV", ascending=False)
    
    if save_csv:
        df_prediction_str_tv.to_csv(path_predictions.joinpath("str_tv.csv"), index="customer_id")
    
    return df_prediction_str_tv

def predict_str_movies(df, model_path, save_csv=False):
    df_no_str_movies = df[df["streaming_movies"] == 0].drop(columns="streaming_movies")

    # ========== loading model and creating predictions ==========
    
    model_str_movies = joblib.load(model_path)
    
    pred_prob = model_str_movies.predict_proba(df_no_str_movies)[:, 1]

    # ========== creating prediction dataframe and exporting as a csv ==========
    
    df_prediction_str_movies = pd.DataFrame({
        "Streaming Movies": pred_prob
    }, index=df_no_str_movies.index).sort_values(by="Streaming Movies", ascending=False)
    
    if save_csv:
        df_prediction_str_movies.to_csv(path_predictions.joinpath("str_movies.csv"), index="customer_id")
    
    return df_prediction_str_movies


predict_fiber(df, path_model_fiber, True)
predict_str_tv(df, path_model_str_tv, True)
predict_str_movies(df, path_model_str_movies, True)



