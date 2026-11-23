# Alert explanations: summary

Surrogate target: RiskEngine.compute_risk risk_score (weighted mean over active agents, agreement-amplified). Split: temporal per region: settings.yaml evaluation.split_date / end_date (same as the method comparison); no shuffling.
Top SHAP features: exact mean |SHAP| over every feature of every alert (coverage.json mean_abs_shap).
Complete: the explanation found history and a z-score for each of its top five features. Provenance (code, inputs, artefact hashes): `outputs/MANIFEST.json`.

| Region | Test window | R² test | R² train | Spearman test | Alerts | Complete | Top 5 SHAP features (mean abs SHAP) |
|---|---|---|---|---|---|---|---|
| hormuz | 2023-07-01 .. 2025-11-21 | 0.528 | 0.960 | 0.754 | 201 | 100.0% | diplomatic_incident_score (0.069), vessel_count (0.047), avg_delay_hours (0.038), military_activity_index (0.033), congestion_index (0.029) |
| bab_el_mandeb | 2023-07-01 .. 2025-11-21 | -0.956 | 0.954 | 0.421 | 602 | 100.0% | diplomatic_incident_score (0.071), avg_delay_hours (0.039), congestion_index (0.027), military_activity_index (0.027), vessel_count (0.020) |
| panama | 2023-07-01 .. 2025-11-20 | 0.637 | 0.939 | 0.793 | 193 | 100.0% | vessel_count (0.107), avg_delay_hours (0.051), congestion_index (0.026), article_volume (0.010), sentiment_score (0.009) |
| malacca | 2023-07-01 .. 2025-11-21 | 0.488 | 0.976 | 0.842 | 311 | 100.0% | vessel_count (0.053), diplomatic_incident_score (0.052), military_activity_index (0.046), avg_delay_hours (0.029), congestion_index (0.016) |

A negative held-out R² means the surrogate predicts the composite worse than its mean on the test window, so its SHAP values explain the surrogate more than the composite: bab_el_mandeb (held-out R² -0.956).
