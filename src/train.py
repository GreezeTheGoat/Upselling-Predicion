from useful_funcs import test_model, get_shap
import pandas as pd
import yaml
import sys
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ========== Opening config file ==========

path_yaml = Path("../src/config.yaml")
try:

    with open(path_yaml, "r") as file:
        config = yaml.safe_load(file)

except FileNotFoundError:
    print("Config file not found")
    sys.exit(1)

# ========== opening dataframe and setting paths ==========

path_preprocessed = Path("../data/processed/telco_preprocessed.csv")
df = pd.read_csv(path_preprocessed, index_col="customer_id")

pd.set_option('future.no_silent_downcasting', True)

# ========== Spliting dataframe ==========

X = df.drop(columns = ["is_fiber"])
y = df["is_fiber"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Using ColumnTransformer on df ==========

num_features = config["features"]["numerical_features"]
ord_features = [config["features"]["ordinal_categoricals"][0]]
nom_features = config["features"]["nominal_categoricals"]
ord_feat_categ = [config["features"]["ordinal_orders"][0]]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("ord", OrdinalEncoder(categories=ord_feat_categ), ord_features),
         ("nom", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nom_features), 
    ],
    remainder="drop"
)

preprocessor.set_output(transform="pandas")

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# ========== Training default model to get best columns and for future comparsion ==========

default_model = lgb.LGBMClassifier(verbose=-1)
default_model.fit(X_train, y_train)
col_imp_ranking = get_shap(default_model, X_train)

n_features = config["features"]["n_features_to_select"]
X_filtered = X_train[col_imp_ranking[:n_features]]

# ========== Training optimized model using optun ==========

best_params = {'learning_rate': 0.05898602410432694,
'num_leaves': 20,
'min_child_samples': 65,
'colsample_bytree': 0.6682096494749166}

optimized_model = lgb.LGBMClassifier(
**best_params
)

optimized_model.fit(X_train[col_imp_ranking[:n_features]], y_train)

# ========== Testing and comparing the optimized model ==========

optimized_results = test_model(
        optimized_model,
        X_test[col_imp_ranking[:n_features]],
        y_test,
        "Optimized Model"
        )

default_results = test_model(default_model, X_test, y_test, "Default Model")

df_final = pd.concat([default_results, optimized_results], axis=1 )

print(df_final)

# ========== Dumping the model ==========

path_model = Path("../models/fiber.joblib")

joblib.dump(optimized_model, path_model)
