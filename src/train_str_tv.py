from useful_funcs import test_model, filter_column
import pandas as pd
import yaml
import sys
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

# ========== Spliting dataframe and formating y axis ==========

X = df.drop(columns = ["streaming_tv"])
y = df["streaming_tv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

quick_map = {"yes": 1, "no": 0}
y_train = y_train.replace(quick_map).astype(int)
y_test = y_test.replace(quick_map).astype(int)

# ========== Creating preprocessor ==========

num_features = config["features"]["numerical_features"]
ord_features = [config["features"]["ordinal_categoricals"][0]]
nom_features = config["features"]["nominal_categoricals"]
ord_feat_categ = [config["features"]["ordinal_orders"][0]]
col_imp_ranking = config["features"]["columns_ord_shap"]

nom_features.remove("streaming_tv")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("ord", OrdinalEncoder(categories=ord_feat_categ), ord_features),
         ("nom", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nom_features), 
    ],
    remainder="drop"
)


preprocessor.set_output(transform="pandas")

# ========== Creating optimized model using parameter from optuna ==========

best_params = {
    "verbose": -1,
    "learning_rate": 0.2,
    "num_leaves": 7,
    "min_child_samples": 10,
    "colsample_bytree": 0.511182,
    "n_estimators": 34,
    "random_state": 42,
    "force_row_wise": True
}

optimized_model = lgb.LGBMClassifier(
**best_params,
)

# ========== Creating optimized pipeline and default too for comparsion ==========

optimized_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("optimized_model", optimized_model)
]
)
optimized_pipeline.fit(X_train, y_train)

default_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("default_model", lgb.LGBMClassifier(verbose=-1))
])
default_pipeline.fit(X_train, y_train)

# ========== Testing and comparing the optimized model ==========

optimized_results = test_model(
        optimized_pipeline,
        X_test,
        y_test,
        "Optimized Model"
        )

default_results = test_model(default_pipeline, X_test, y_test, "Default Model")

df_final = pd.concat([default_results, optimized_results], axis=1 )

# ========== Dumping the model ==========

path_model = Path(config["paths"]["models"]).joinpath("str_tv.pkl")

joblib.dump(optimized_pipeline, path_model)

print(df_final)
