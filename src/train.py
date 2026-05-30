import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def model_training():

    try:

        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        print("---------- config.yaml file not found ----------")
        sys.exit(1)

    path_processed = config["paths"]["processed"]
    path_model = config["paths"]["model"]


    optimal_columns = config["features"]["optimized_columns"]
    if len(optimal_columns) != 13:
        print("---------- You need to have exactly 13 columns for optimal model ----------")

    # ---------- Opening all the training data frames ----------

    try:

        X_train = pd.read_csv(path_processed + "X_train.csv")
        y_train = pd.read_csv(path_processed + "y_train.csv").squeeze("columns")
        X_test = pd.read_csv(path_processed + "X_test.csv")
        y_test = pd.read_csv(path_processed + "y_test.csv").squeeze("columns")

    except:
        print("---------- You have to preprocess raw data in order to train the model ----------")
        sys.exit(1)

    # ---------- Optimizing the data frame based on SFS and RFE ----------

    X_train_opt = X_train[optimal_columns]
    X_test_opt = X_test[optimal_columns]

    # ---------- Training the model ----------

    print("---------- Training Model ----------")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 12, 16, None],          
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='precision',
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train_opt, y_train)

    model = grid_search.best_estimator_

    print("---------- Model trained with success ----------")

    # ---------- Diagnosing the model ----------

    prediction = model.predict(X_test_opt)
    print("---------- Accuracy score -----------")

    print(f"{accuracy_score(y_test, prediction) * 100:.2f}%\n")

    print("---------- Classification report ----------")
    print(classification_report(y_test, prediction))

    # ---------- Exporting model as .pkl ----------

    joblib.dump(model, path_model)

    print("---------- Model file created with success ----------")

if __name__ == "__main__":
    model_training()
