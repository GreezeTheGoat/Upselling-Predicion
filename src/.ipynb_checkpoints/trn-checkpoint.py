from usefull_funcs import test_model, get_shap
import pandas as pd
import yaml
import sys
import yaml
import lightgbm as lgb
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ========== ==========

path_yaml = Path("../src/config.yaml")
try:

    with open(path_yaml, "r") as file:
        config = yaml.safe_load(file)

except FileNotFoundError:
    print("Config file not found")
    sys.exit(1)

# ========== ==========

path_preprocessed = Path("../data/processed/telco_preprocessed.csv")
df = pd.read_csv(path_preprocessed)

pd.set_option('future.no_silent_downcasting', True)

# ========== ==========

id_col = config["mapping"]["customer_id"]
if id_col in df.columns:
    df.set_index(id_col, inplace=True)

# ========== ==========

X = df.drop(columns = ["is_fiber"])
y = df["is_fiber"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X_train)

# ========== ==========

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

# ========== ==========

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

# ========== ==========

default_model = lgb.LGBMClassifier()
default_model.fit(X_train, y_train)
col_imp_ranking = get_shap(default_model, X_train)

n_features = 18
X_filtered = X_train[col_imp_ranking[:n_features]]
# ========== ==========

best_params = {'learning_rate': 0.05898602410432694,
'num_leaves': 20,
'min_child_samples': 65,
'colsample_bytree': 0.6682096494749166}
lgbm_clf = lgb.LGBMClassifier(
**best_params
)

lgbm_clf.fit(X_train[col_imp_ranking[:n_features]], y_train)

# ========== ==========

optimized_results = test_model(
        lgbm_clf,
        X_test[col_imp_ranking[:n_features]],
        y_test,
        "Optimized Model"
        )

# ========== ==========

print(df)

# ========== ==========
# ========== ==========
