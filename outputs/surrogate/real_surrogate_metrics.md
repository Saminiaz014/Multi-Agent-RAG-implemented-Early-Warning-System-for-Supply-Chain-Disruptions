# Real-data SHAP surrogate

Target: RiskEngine.compute_risk risk_score (weighted mean over active agents, agreement-amplified). Model: RandomForestRegressor(n_estimators=100, random_state=42). Split: temporal per region: settings.yaml evaluation.split_date / end_date (same as the method comparison); no shuffling.

Missing features: per-region: features the region does not supply are dropped; pooled: features present in at least one region are kept and left NaN where absent (RandomForest native NaN handling); features absent from every region are dropped. No zero-filling, no scaler, no imputer.

| Model | Features | R² train | R² test | MAE test | Spearman test | Test days ≥ medium | MAE on those | Top-1 agent agreement | Top-3 agent agreement |
|---|---|---|---|---|---|---|---|---|---|
| hormuz | 15 | 0.960 | 0.528 | 0.0680 | 0.754 | 28.7% (251) | 0.1103 | 72.5% | 95.6% |
| bab_el_mandeb | 15 | 0.954 | -0.956 | 0.1620 | 0.421 | 91.4% (800) | 0.1702 | 36.8% | 90.0% |
| panama | 15 | 0.939 | 0.637 | 0.0692 | 0.793 | 26.2% (229) | 0.0841 | 98.3% | 98.7% |
| malacca | 15 | 0.976 | 0.488 | 0.0564 | 0.842 | 14.1% (123) | 0.1382 | 43.1% | 91.9% |
| pooled | 15 | 0.965 | 0.598 | 0.0831 | 0.816 | 40.1% (1403) | 0.1196 | 52.9% | 93.9% |

## For comparison: the deployed synthetic-trained surrogate on the same real windows

| Model | Features | R² train | R² test | MAE test | Spearman test | Test days ≥ medium | MAE on those | Top-1 agent agreement | Top-3 agent agreement |
|---|---|---|---|---|---|---|---|---|---|
| hormuz | 17 | 0.012 | -0.132 | 0.1031 | 0.246 | 28.7% (251) | 0.2136 | 36.3% | 53.8% |
| bab_el_mandeb | 17 | -0.036 | -2.621 | 0.2319 | 0.243 | 91.4% (800) | 0.2500 | 9.9% | 87.0% |
| panama | 17 | -1.859 | -0.838 | 0.1648 | -0.003 | 26.2% (229) | 0.0685 | 0.4% | 5.2% |
| malacca | 17 | -1.570 | 0.118 | 0.0799 | 0.242 | 14.1% (123) | 0.1673 | 14.6% | 39.0% |
