# Real-data SHAP surrogate

Target: RiskEngine.compute_risk risk_score (weighted mean over active agents, agreement-amplified). Model: RandomForestRegressor(n_estimators=100, random_state=42). Split: temporal per region: settings.yaml evaluation.split_date / end_date (same as the method comparison); no shuffling.

Missing features: per-region: features the region does not supply are dropped; pooled: features present in at least one region are kept and left NaN where absent (RandomForest native NaN handling); features absent from every region are dropped. No zero-filling, no scaler, no imputer.

| Model | Features | R² train | R² test | MAE test | Spearman test | Test days ≥ medium | MAE on those | Top-1 agent agreement | Top-3 agent agreement |
|---|---|---|---|---|---|---|---|---|---|
| hormuz | 15 | 0.963 | 0.496 | 0.0697 | 0.720 | 32.8% (287) | 0.1057 | 80.1% | 96.9% |
| bab_el_mandeb | 11 | 0.954 | -1.376 | 0.1705 | 0.314 | 91.1% (797) | 0.1804 | 40.9% | 93.7% |
| panama | 12 | 0.937 | 0.637 | 0.0691 | 0.791 | 26.2% (229) | 0.0841 | 98.3% | 98.7% |
| malacca | 12 | 0.977 | 0.557 | 0.0516 | 0.877 | 14.1% (123) | 0.1323 | 61.8% | 98.4% |
| pooled | 15 | 0.967 | 0.575 | 0.0837 | 0.817 | 41.0% (1436) | 0.1238 | 54.4% | 97.8% |

## For comparison: the deployed synthetic-trained surrogate on the same real windows

| Model | Features | R² train | R² test | MAE test | Spearman test | Test days ≥ medium | MAE on those | Top-1 agent agreement | Top-3 agent agreement |
|---|---|---|---|---|---|---|---|---|---|
| hormuz | 17 | 0.050 | -0.308 | 0.1105 | 0.251 | 32.8% (287) | 0.2223 | 42.5% | 63.1% |
| bab_el_mandeb | 17 | -0.026 | -2.885 | 0.2314 | 0.243 | 91.1% (797) | 0.2493 | 5.8% | 86.6% |
| panama | 17 | -1.859 | -0.838 | 0.1648 | -0.003 | 26.2% (229) | 0.0685 | 0.4% | 5.2% |
| malacca | 17 | -1.570 | 0.118 | 0.0799 | 0.242 | 14.1% (123) | 0.1673 | 14.6% | 39.0% |
