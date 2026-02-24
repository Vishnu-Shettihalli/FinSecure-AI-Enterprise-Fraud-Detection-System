import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_confusion_matrix(model, X_test, y_test, model_name):
    if model_name == "IsolationForest":
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name):
    if model_name == "IsolationForest":
        return  # ROC not meaningful for this version

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_score = roc_auc_score(y_test, y_proba)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_score:.3f})")


def compare_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8,6))

    for name, model in models.items():
        if name != "IsolationForest":
            plot_roc_curve(model, X_test, y_test, name)

    plt.plot([0,1], [0,1], linestyle='--')
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    import numpy as np


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8,6))
    sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices]
    )

    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()