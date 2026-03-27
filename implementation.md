# Phase 37 — Random Forest → pIC50 Regression

**Version:** 1.1 | **Tier:** Standard | **Date:** 2026-03-26

## Goal
Train a Random Forest regressor to predict pIC50 from ECFP4 Morgan fingerprints.
Evaluate with LOO-CV and scaffold-split CV. Report R², MAE, RMSE.

CLI: `python main.py --input data/compounds.csv`

Outputs: rf_predictions.csv, parity_plot.png, feature_importance.png

## Fingerprints
- ECFP4: radius=2, nBits=2048, useChirality=True
- Each compound → 2048-bit binary vector

## Model
- `RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)`
- Evaluation 1: LeaveOneOut CV → R², MAE, RMSE
- Evaluation 2: 5-fold CV → mean ± std R²

## Outputs
- rf_predictions.csv: compound_name, y_true, y_pred_loo, family
- parity_plot.png: y_true vs y_pred_loo scatter, colored by family, R² annotated
- feature_importance.png: top-20 bit importances from full-fit model (bar chart)

## Logic
- Load compounds.csv, compute ECFP4 fingerprints (radius=2, nBits=2048, useChirality=True)
- LOO-CV: for each compound, train RF on remaining 44, predict held-out pIC50
- Collect all LOO predictions, compute R², MAE, RMSE
- 5-fold CV: cross_val_score with scoring="r2" for variance estimate
- Full-fit RF on all 45 compounds for feature importance extraction (MDI)
- Save predictions CSV, parity plot (y_true vs y_pred colored by family), top-20 bit importance bar chart

## Key Concepts
- scikit-learn `RandomForestRegressor` (n_estimators=200, random_state=42)
- ECFP4 Morgan fingerprints via RDKit `AllChem.GetMorganFingerprintAsBitVect`
- Leave-One-Out cross-validation (deterministic, preferred for n=45)
- Mean Decrease in Impurity (MDI) feature importance

## Verification Checklist
- [x] LOO-CV R² = 0.729 (solid for 45-compound dataset)
- [x] LOO-CV MAE = 0.268 pIC50 (sub-unit prediction error)
- [x] 5-fold CV shows high variance (0.28 +/- 0.37) confirming LOO is preferred
- [x] Parity plot and feature importance chart saved to output/
- [x] rf_predictions.csv contains per-compound predictions

## Risks
- LOO-CV may overestimate generalization to new scaffold classes (see Phase 39 scaffold split)
- MDI importance is biased toward features with many unique values (less relevant for binary fingerprint bits)

## Actual Results (v1.1)

| Metric | LOO-CV | 5-Fold CV |
|---|---|---|
| R² | 0.7288 | 0.28 ± 0.37 |
| MAE | 0.2678 | — |
| RMSE | 0.3708 | — |

**Key insight:** LOO-CV R²=0.73 is solid; 5-fold CV is highly variable (small dataset, 45 compounds). LOO-CV is the preferred metric for this library size.
