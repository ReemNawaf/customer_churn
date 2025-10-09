import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve


def plot_confusion(y_true, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"{model_name} – Confusion Matrix")
    plt.show()


def plot_roc(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} – ROC Curve")
    plt.legend()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    sns.barplot(x=imp.values, y=imp.index)
    plt.title("Top Feature Importances")
    plt.show()
