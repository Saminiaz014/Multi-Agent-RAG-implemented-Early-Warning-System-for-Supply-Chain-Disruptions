# Alert explanations: summary

Surrogate target: RiskEngine.compute_risk risk_score (weighted mean over active agents, agreement-amplified). Split: temporal per region: settings.yaml evaluation.split_date / end_date (same as the method comparison); no shuffling.
Top SHAP features: exact mean |SHAP| over every feature of every alert (coverage.json mean_abs_shap).
Complete: the explanation found history and a z-score for each of its top five features. Provenance (code, inputs, artefact hashes): `outputs/MANIFEST.json`.

| Region | Test window | R² test | R² train | Spearman test | Alerts | Complete | Top 5 SHAP features (mean abs SHAP) |
|---|---|---|---|---|---|---|---|
| hormuz | 2023-07-01 .. 2025-11-21 | 0.496 | 0.963 | 0.720 | 214 | 100.0% | diplomatic_incident_score (0.062), vessel_count (0.045), military_activity_index (0.042), avg_delay_hours (0.036), congestion_index (0.036) |
| bab_el_mandeb | 2023-07-01 .. 2025-11-21 | -1.376 | 0.954 | 0.314 | 626 | 100.0% | diplomatic_incident_score (0.055), congestion_index (0.035), avg_delay_hours (0.033), military_activity_index (0.021), vessel_count (0.018) |
| panama | 2023-07-01 .. 2025-11-20 | 0.637 | 0.937 | 0.791 | 193 | 100.0% | vessel_count (0.107), avg_delay_hours (0.050), congestion_index (0.026), article_volume (0.010), sentiment_score (0.009) |
| malacca | 2023-07-01 .. 2025-11-21 | 0.557 | 0.977 | 0.877 | 311 | 100.0% | vessel_count (0.078), diplomatic_incident_score (0.051), military_activity_index (0.045), avg_delay_hours (0.027), congestion_index (0.019) |

A negative held-out R² means the surrogate predicts the composite worse than its mean on the test window, so its SHAP values explain the surrogate more than the composite: bab_el_mandeb (held-out R² -1.376).
