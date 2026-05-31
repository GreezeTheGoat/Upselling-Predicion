import pandas as pd
import joblib
import yaml
import sys
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def model_training():
    """
    Gather all processed data, train optimized RFC model based on that data,
    create .pkl model file

    """

    try:

        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        logging.error("---------- config.yaml file not found ----------")
        sys.exit(1)

    path_processed = Path(config["paths"]["processed"])
    path_model = Path(config["paths"]["model"])


    optimal_columns = config["features"]["optimized_columns"]
    if len(optimal_columns) != 13:
        logging.info("---------- You need to have exactly 13 columns for optimal model ----------")

    # ---------- Opening all the training data frames ----------

    try:

        X_train = pd.read_csv(path_processed.joinpath("X_train.csv"))
        y_train = pd.read_csv(path_processed.joinpath("y_train.csv")).squeeze("columns")
        X_test = pd.read_csv(path_processed.joinpath("X_test.csv"))
        y_test = pd.read_csv(path_processed.joinpath("y_test.csv")).squeeze("columns")

    except:
        logging.error("---------- You have to preprocess raw data in order to train the model ----------")
        sys.exit(1)

    # ---------- Optimizing the data frame based on SFS and RFE ----------

    X_train_opt = X_train[optimal_columns]
    X_test_opt = X_test[optimal_columns]

    # ---------- Training the model ----------

    logging.info("---------- Training Model ----------")

    model = RandomForestClassifier(random_state=42, max_depth=12, max_features= "sqrt", min_samples_split= 10, n_estimators=200)

    model.fit(X_train_opt, y_train)
    
    logging.info("---------- Model trained with success ----------")

    # ---------- Diagnosing the model ----------

    prediction = model.predict(X_test_opt)
    logging.info("---------- Accuracy score -----------")

    logging.info(f"{accuracy_score(y_test, prediction) * 100:.2f}%\n")

    logging.info("---------- Classification report ----------")
    logging.info(classification_report(y_test, prediction))

    # ---------- Exporting model as .pkl ----------

    joblib.dump(model, path_model)

    logging.info("---------- Model file created with success ----------")

if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
    ) 
    model_training()
