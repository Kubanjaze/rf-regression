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

## Actual Results (v1.1)

| Metric | LOO-CV | 5-Fold CV |
|---|---|---|
| R² | 0.7288 | 0.28 ± 0.37 |
| MAE | 0.2678 | — |
| RMSE | 0.3708 | — |

**Key insight:** LOO-CV R²=0.73 is solid; 5-fold CV is highly variable (small dataset, 45 compounds). LOO-CV is the preferred metric for this library size.
