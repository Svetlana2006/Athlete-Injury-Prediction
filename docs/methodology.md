# Methodology and safety notes

## Prediction target

For every athlete-day, the supervised target is `1` when a recorded injury starts in the next seven days, excluding information recorded after that prediction time. The horizon is configurable through `RISK_WINDOW_DAYS`.

## Leakage controls

- Sort data by athlete and observation date before any rolling computation.
- Compute features using observations on or before the prediction date only.
- Split by time: early dates train the model and later dates evaluate it.
- Never use injury status, return-to-play data, or future GPS values as predictors at the earlier prediction time.

## Metrics

Injuries are rare; accuracy is therefore not a useful primary metric. Report PR-AUC, precision, recall, F1, number of positives, and a confusion matrix at a stated threshold. Compare the trained model with the transparent rule-based score.

## Interpretability

The API always returns contributing factors. The initial production-safe fallback is a documented rule score. If a model has been trained, its probability is combined with—not substituted for—those visible workload/recovery signals.

## Cricket adaptation

The shared `Observation` model supports cricket-specific fields through `sport_metrics` (for example balls bowled, overs, spells, and match frequency). No clinical claims should be made until it is trained and evaluated on suitable cricket injury-labelled data.
