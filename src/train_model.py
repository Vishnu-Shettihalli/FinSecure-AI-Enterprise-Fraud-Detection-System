import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=1,  # already balanced using SMOTE
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train):
    model = IsolationForest(
        n_estimators=100,
        contamination=0.0017,  # approx fraud ratio
        random_state=42
    )
    model.fit(X_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    if model_name == "IsolationForest":
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]
    else:
        y_pred = model.predict(X_test)

    print(f"\n--- {model_name} Evaluation ---")
    print(classification_report(y_test, y_pred))

    if model_name != "IsolationForest":
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_score = roc_auc_score(y_test, y_proba)
        print("ROC-AUC Score:", roc_score)