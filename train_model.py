import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split

from src.data_loader import load_data, get_feature_names, get_class_info
from src.preprocessing import split_features_target, fit_scaler, apply_scaler, oversample_minority
from src.model import train, evaluate_cv, predict
from src.evaluate import (
    print_report, plot_confusion_matrix, plot_roc_and_pr_curves,
    plot_feature_importances, plot_learning_curve,
    plot_class_imbalance, plot_feature_selection_impact,
)

DATA_PATH   = "data/ckd_data.csv"
MODEL_PATH  = "outputs/model.pkl"
OUTPUT_DIR  = "outputs"
PLOTS_DIR   = "outputs/plots"


def main():
    print("=" * 55)
    print("CKD Prediction — Training Pipeline")
    print(f"Features used: {len(get_feature_names())} (selected from 51)")
    print("=" * 55)

    # 1. Load
    print("\n[1/6] Loading data...")
    df = load_data(DATA_PATH)
    print(f"  {len(df)} samples, {df.shape[1]-1} total features, using {len(get_feature_names())}")
    get_class_info(df)
    plot_class_imbalance(df["Diagnosis"].values, save_dir=PLOTS_DIR)

    # 2. Preprocess
    print("\n[2/6] Preprocessing (scale + oversample)...")
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_sc, _ = fit_scaler(X_tr, save_dir=OUTPUT_DIR)
    X_te_sc     = apply_scaler(X_te, save_dir=OUTPUT_DIR)
    X_tr_bal, y_tr_bal = oversample_minority(X_tr_sc, y_tr)
    print(f"  Train (balanced): {len(X_tr_bal)} | Test: {len(X_te)}")

    # 3. Cross-validate
    print("\n[3/6] 10-fold stratified CV (on balanced train)...")
    evaluate_cv(X_tr_bal, y_tr_bal)

    # 4. Train
    print("\n[4/6] Training final model...")
    model = train(X_tr_bal, y_tr_bal, save_path=MODEL_PATH)
    tr_acc = model.score(X_tr_sc, y_tr)
    te_acc = model.score(X_te_sc, y_te)
    print(f"  Train acc : {tr_acc:.4f}")
    print(f"  Test  acc : {te_acc:.4f}  (gap: {tr_acc-te_acc:.4f})")

    # 5. Evaluate
    print("\n[5/6] Evaluating on test set...")
    y_pred, y_proba = predict(model, X_te_sc)
    print_report(y_te, y_pred)
    auc, ap = plot_roc_and_pr_curves(y_te, y_proba, save_dir=PLOTS_DIR)
    print(f"  ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")
    plot_confusion_matrix(y_te, y_pred, save_dir=PLOTS_DIR)
    plot_feature_importances(model, get_feature_names(), save_dir=PLOTS_DIR)

    # 6. Diagnostic plots
    print("\n[6/6] Diagnostic plots...")
    X_all_sc, _ = fit_scaler(X, save_dir=OUTPUT_DIR)
    plot_learning_curve(X_all_sc, y, save_dir=PLOTS_DIR)
    plot_feature_selection_impact(save_dir=PLOTS_DIR)

    print("\nAll done. Outputs in outputs/")


if __name__ == "__main__":
    main()
