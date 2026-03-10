from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import os


def build_model():
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )


def train(X_train, y_train, save_path: str = "outputs/model.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    return model


def load_model(model_path: str = "outputs/model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}. Run train_model.py first.")
    return joblib.load(model_path)


def evaluate_cv(X, y):
    model = build_model()
    skf   = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    res   = cross_validate(model, X, y, cv=skf,
                           scoring=["roc_auc", "f1_macro", "recall_macro"])
    print(f"  ROC-AUC    : {res['test_roc_auc'].mean():.4f} ± {res['test_roc_auc'].std():.4f}")
    print(f"  F1 (macro) : {res['test_f1_macro'].mean():.4f} ± {res['test_f1_macro'].std():.4f}")
    print(f"  Recall(mac): {res['test_recall_macro'].mean():.4f} ± {res['test_recall_macro'].std():.4f}")
    return res


def predict(model, X):
    return model.predict(X), model.predict_proba(X)
