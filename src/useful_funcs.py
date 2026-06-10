import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score, accuracy_score,  f1_score

def test_model(model, X_t, y_t, name="Model", pos_label=1):
    prob_indices = list(model.classes_)
    pos_idx = prob_indices.index(pos_label)
    pred_prob = model.predict_proba(X_t)[:, pos_idx]
    
    pred_acc = model.predict(X_t)
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_t, pred_prob),
        "PR-AUC": average_precision_score(y_t, pred_prob, pos_label=pos_label),
        "Log loss": log_loss(y_t, model.predict_proba(X_t)),
        "Brier Score": brier_score_loss(y_t, pred_prob, pos_label=pos_label),
        "Accuracy": accuracy_score(y_t, pred_acc),
        "F1-Score": f1_score(y_t, pred_acc, pos_label=pos_label)
        }

    df_report = pd.DataFrame.from_dict(metrics, orient='index', columns=[name])
    return df_report


def get_shap(model, X: pd.DataFrame, plot: bool =False) -> list:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X, check_additivity=False)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    feature_ranking = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False)

    col_ranking = feature_ranking["feature"].tolist()
    
    if plot == True:
        plt.figure(figsize=(10,6))
        shap.plots.beeswarm(shap_values)
        plt.show()

    return col_ranking

def filter_column(X, columns):
    return X[columns]
