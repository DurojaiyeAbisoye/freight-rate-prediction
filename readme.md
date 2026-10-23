# Freight Rate Prediction Challenge

Predicts `posted_rate` for freight loads. Final model: LightGBM (log1p target), tuned with Optuna.
Assessment instructions: `freight-rate-ml-assessment-1-.pdf`.

## Repository layout

```
.
├── data/
│   ├── train-test.csv                      # labeled development data (48,000 loads)
│   ├── validation.csv                      # 12,000 loads to predict
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv           # 31 fixed December rows
├── freight_rate_pipeline_ordered.ipynb     # full pipeline: EDA, cleaning, features, models, tuning, predictions
├── output/                                 # created by the notebook
│   ├── validation_predictions.csv          # final predictions (load_id, predicted_rate)
│   ├── december-chart-inputs.csv           # December predictions
│   └── final_model.joblib, feature_pipeline.joblib, equip_median_wt.joblib
├── score.py                                # provided scorer
├── scorer_results/candidate_december.png   # chart produced by score.py
├── requirements.txt                        # scorer requirements
├── requirements-notebook.txt               # notebook requirements
└── readme.md
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebook.txt
```

## Run

1. Put the four CSV files in `data/` (names as in the layout above).
2. Open and run the notebook top to bottom:

   ```bash
   jupyter lab freight_rate_pipeline_ordered.ipynb
   ```

   It cleans the data, builds features, compares baselines, tunes LightGBM, refits on all
   labeled data, and writes everything to `output/`.
   Note: the Optuna step (5 trials, 5 time-series folds each) takes several minutes per trial.
3. Score the predictions (also run in the last notebook cells):

   ```bash
   python score.py --predictions output/validation_predictions.csv --december-predictions output/december-chart-inputs.csv
   ```

   The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Approach in brief

- **Cleaning:** negative weights flipped to positive; missing weight filled with per-equipment training median; missing `market_index` filled with training median.
- **Split:** time-based. Last 31 days of the labeled data (Oct 2025) held out; tuning uses `TimeSeriesSplit`.
- **Model:** LightGBM on `log1p(posted_rate)`. Holdout RMSE 648.67, R2 0.820 (best statistical baseline: RMSE 683.87).
- **December chart:** missing inputs rebuilt (coordinates from training lookup, daily median `quote_signal` from validation, `market_index` imputed).
