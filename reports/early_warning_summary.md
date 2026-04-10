# Early Warning Summary Report

## Executive summary

- Best supervised failure classifier: **XGBoost** with F1 **0.87**, AP **0.95**, and AUROC **1.00**.
- Best unsupervised anomaly detector: **One-Class SVM** with F1 **0.77**, AP **0.86**, and AUROC **1.00**.
- Median early warning lead time for the top classifier: **30 cycles** before the 30-cycle failure horizon.
- Most informative signals: **sensor_4, sensor_11, sensor_3, sensor_2, sensor_21**.

## Operational interpretation

- Sensors with the strongest degradation patterns consistently drift away from their healthy operating baseline as engines approach failure.
- The supervised models are better suited for explicit maintenance triage when historical failure labels exist.
- The anomaly detectors still add value for cold-start monitoring because they flag departures from the healthy manifold without needing failure labels.

## Recommendation

- Use the supervised classifier for scheduled maintenance planning.
- Use the anomaly model as an always-on shadow monitor to escalate unusual behavior sooner.
- Review the top-signal trend plots during root-cause analysis to explain why the alert fired.
