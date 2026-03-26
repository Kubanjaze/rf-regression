# rf-regression — Phase 37

Random Forest regression on ECFP4 fingerprints to predict pIC50.
Evaluated with LOO-CV and 5-fold CV.

## Usage

```bash
PYTHONUTF8=1 python main.py --input data/compounds.csv
```

## Outputs

| File | Description |
|---|---|
| `output/rf_predictions.csv` | LOO-CV predictions per compound |
| `output/parity_plot.png` | Observed vs predicted pIC50, colored by family |
| `output/feature_importance.png` | Top-20 ECFP4 bit importances |

## Fingerprints
ECFP4: radius=2, nBits=2048, useChirality=True
