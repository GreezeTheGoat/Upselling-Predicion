import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def model_training():

    # ---------- Opening all the training data frames ----------

    path_processed = "../data/processed/"
    X_train = pd.read_csv(f"{path_processed}X_train.csv")
    y_train = pd.read_csv(f"{path_processed}y_train.csv").squeeze("columns")
    X_test = pd.read_csv(f"{path_processed}X_test.csv")
    y_test = pd.read_csv(f"{path_processed}y_test.csv").squeeze("columns")

    # ---------- Optimizing the data frame based on SFS and RFE ----------

    rfc_optimal_columns = ['CLTV',
     'Tenure Months',
     'Total Extra Services',
     'Payment Method',
     'Multiple Lines',
     'Contract',
     'Streaming TV',
     'Phone Service',
     'Paperless Billing',
     'Streaming Movies',
     'Senior Citizen',
     'Tech Support',
     'Online Security']

    X_train_opt = X_train[rfc_optimal_columns]
    X_test_opt = X_test[rfc_optimal_columns]

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

    joblib.dump(model, '../models/random_forest_final.pkl')

    print("---------- Model file created with success ----------")

if __name__ == "__main__":
    model_training()
