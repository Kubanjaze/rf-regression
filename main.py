import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RDLogger.DisableLog("rdApp.*")

FAMILY_COLORS = {"benz": "#4C72B0", "naph": "#DD8452", "ind": "#55A868",
                 "quin": "#C44E52", "pyr": "#8172B2", "bzim": "#937860", "other": "#808080"}

def load_compounds(path):
    df = pd.read_csv(path)
    records, n_bad = [], 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None: n_bad += 1; continue
        try:
            pic50 = float(row["pic50"])
        except (KeyError, ValueError):
            continue
        if np.isnan(pic50): continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        fam = str(row["compound_name"]).split("_")[0]
        records.append({"compound_name": str(row["compound_name"]),
                        "family": fam if fam in FAMILY_COLORS else "other",
                        "pic50": pic50, "fp": list(fp)})
    print(f"  {len(records)} valid ({n_bad} skipped)")
    return pd.DataFrame(records)

def plot_parity(y_true, y_pred, families, r2, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for fam, color in FAMILY_COLORS.items():
        mask = [f == fam for f in families]
        if any(mask):
            ax.scatter(np.array(y_true)[mask], np.array(y_pred)[mask],
                       color=color, label=fam, s=50, alpha=0.85, edgecolors="white", linewidth=0.5)
    lo, hi = min(y_true) - 0.2, max(y_true) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Observed pIC50", fontsize=11)
    ax.set_ylabel("Predicted pIC50 (LOO-CV)", fontsize=11)
    ax.set_title(f"Random Forest Regression (LOO-CV, R²={r2:.3f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, top_n, output_path):
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(top_n), imp[top_idx], color="#4C72B0", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"bit{i}" for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature Importance (MDI)", fontsize=11)
    ax.set_title(f"Top {top_n} ECFP4 Bit Importances", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_compounds(args.input)
    X = np.array(df["fp"].tolist())
    y = df["pic50"].values

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    # LOO-CV
    print("Running LOO-CV...")
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        rf.fit(X[train_idx], y[train_idx])
        y_pred_loo[test_idx] = rf.predict(X[test_idx])
    r2_loo = r2_score(y, y_pred_loo)
    mae_loo = mean_absolute_error(y, y_pred_loo)
    rmse_loo = np.sqrt(mean_squared_error(y, y_pred_loo))

    # 5-fold CV
    cv_r2 = cross_val_score(rf, X, y, cv=5, scoring="r2")

    # Full-fit for feature importance
    rf.fit(X, y)

    # Save predictions
    pred_df = pd.DataFrame({"compound_name": df["compound_name"],
                            "family": df["family"],
                            "y_true": y, "y_pred_loo": y_pred_loo})
    pred_df.to_csv(os.path.join(args.output_dir, "rf_predictions.csv"), index=False)
    print(f"Saved: {args.output_dir}/rf_predictions.csv")

    plot_parity(y, y_pred_loo, df["family"].tolist(), r2_loo,
                os.path.join(args.output_dir, "parity_plot.png"))
    print(f"Saved: {args.output_dir}/parity_plot.png")

    plot_feature_importance(rf, 20, os.path.join(args.output_dir, "feature_importance.png"))
    print(f"Saved: {args.output_dir}/feature_importance.png")

    print(f"\n--- LOO-CV Results ---")
    print(f"  R²:   {r2_loo:.4f}")
    print(f"  MAE:  {mae_loo:.4f}")
    print(f"  RMSE: {rmse_loo:.4f}")
    print(f"\n--- 5-Fold CV R² ---")
    print(f"  Mean: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  Folds: {[round(x,3) for x in cv_r2]}")
    print("\nDone.")

if __name__ == "__main__":
    main()
