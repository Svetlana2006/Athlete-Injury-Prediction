# Athlete Injury-Risk Assessment

An exploratory athlete-monitoring project built around the SoccerMon football dataset. It combines workload, wellness, GPS summaries, and injury records to study whether recent athlete observations can help identify an elevated risk of a recorded injury in the following seven days.

The repository includes both an interpretable rule-based risk-assessment API and experimental machine-learning pipelines. The Transformer experiments described below are research experiments; their scores should not be read as evidence of clinical validity or as a guarantee that an injury will occur.

> **Safety:** This project is decision support and a research prototype, not a medical diagnosis or a substitute for clinical judgment.

## The project story

The project developed through a series of experiments:

1. **Bring the athlete data together.** SoccerMon provides training-load, wellness, injury, illness, and game-performance records, plus summarized GPS session data. The aim was to combine these signals because workload, recovery, and injury history may be more informative together than in isolation.
2. **Try engineered features and neural networks.** The pipeline assembled daily athlete observations and included derived workload and wellness features. Early neural-network attempts were described as poor, but the repository does not preserve metrics for the original experiment, so its performance cannot be quantified here.
3. **Try AutoGluon.** Automated model selection was run with engineered and raw feature sets. The saved results show different performance for those particular runs; they do not prove that AutoGluon or one feature representation is universally better.
4. **Compare neural-network architectures.** A benchmark evaluated ten architectures on raw and engineered feature modes. The Transformer had the highest ROC-AUC among the raw-feature models in that benchmark, while another model had higher PR-AUC.
5. **Compare losses for the Transformer.** The current training script supports binary cross-entropy (BCE) and Focal Loss. Saved runs show Focal Loss with `alpha=0.25` and `gamma=2` had higher ROC-AUC than the saved BCE run, but lower PR-AUC. That is a trade-off in these runs, not evidence that Focal Loss is an overall improvement.

In the current neural-network pipeline, “raw” means selected sensor/workload/wellness columns from the processed daily-observation table, not untouched source files. Values are imputed and standardized, and each prediction uses a 14-day input sequence. The current Transformer input has 22 selected columns; it does not consume every source signal (for example, raw accelerometer/gyroscope streams or heart-rate readings).

## Experiment results

These are the figures that could be recovered from checked-in result artifacts. ROC-AUC and PR-AUC are unitless; PR-AUC is reported by the project’s scripts as average precision. Values are rounded to four decimal places.

| Experiment | ROC-AUC | PR-AUC | Result source / interpretation |
|---|---:|---:|---|
| Processed/engineered columns + neural network | 0.8564 | 0.0105 | Transformer entry in the 10-epoch engineered-feature architecture benchmark; this is not a recorded score for the earlier, separate neural-network attempt. |
| Processed/engineered columns + AutoGluon | 0.7291 | 0.0193 | Saved engineered-feature AutoGluon result. |
| Raw columns + best non-Transformer neural network | 0.8912 | 0.0206 | GRU, the raw-feature ROC-AUC leader in the architecture benchmark. CNN-GRU had the highest raw-feature PR-AUC (0.0401) but a lower ROC-AUC (0.8196), so “best” depends on the metric. |
| Raw columns + Transformer + BCE | 0.9116 | 0.0281 | Saved BCE run; chronological cutoff `2021-01-01`. |
| Raw columns + Transformer + Focal Loss (`alpha=0.25`, `gamma=2`) | 0.9210 | 0.0227 | Highest saved matching Focal Loss run; chronological cutoff `2021-01-01`. A later saved repeat scored 0.9191 ROC-AUC and 0.0226 PR-AUC. |

Additional AutoGluon result: the saved raw-feature run reports ROC-AUC **0.8785** and PR-AUC **0.0265**.

### How to read these numbers

- The neural-network architecture benchmark used 10 epochs per model. The saved BCE and Focal Loss training runs used a maximum of 50 epochs with early stopping. These are not identical training procedures.
- The architecture benchmark’s raw-feature GRU and CNN-GRU results are reported separately because the top model changes with the metric.
- The AutoGluon artifacts do not record enough configuration to establish a fully controlled, apples-to-apples comparison with every neural-network run.
- The figures describe saved experiments on this dataset and split. They are not external validation, and no claim is made here that the model generalizes to other teams, sports, or clinical settings.

## Prediction target and methodology

- **Target:** `injury_next_7d` is positive when an injury is recorded as starting in the next seven days. The data pipeline excludes the injury day and two days after it from supervised labels to avoid using observations affected by the injury.
- **Input sequence:** The Transformer receives 14 days of selected daily features for an athlete and predicts the target associated with the sequence’s final day.
- **Preprocessing:** The training pipeline fits median imputation and standardization on pre-cutoff rows, then transforms the later rows.
- **Time split:** The saved neural-network training runs and architecture benchmark use a chronological cutoff of `2021-01-01`.
- **Evaluation caveat:** In `app/ml/train.py`, the post-cutoff loader is used to evaluate each epoch, select the best checkpoint, and report final metrics. It is therefore a model-selection holdout, not an untouched final test set. Repeatedly choosing the best epoch against it can make the reported scores optimistic. A stronger evaluation should reserve a separate final time period that is not used for checkpoint selection or tuning.
- **Rare events:** Injury labels are highly imbalanced. Accuracy alone can be misleading; consider PR-AUC, the number of positive examples, recall/precision at a stated threshold, and top-k ranking metrics together.

## What is in the repository

- FastAPI backend for athlete, observation, injury, and risk-assessment APIs
- Interpretable workload/wellness risk service with contributing-factor explanations
- SoccerMon data pipeline and sequential dataset builder
- Neural-network architecture benchmark and Transformer trainer
- Streamlit dashboard
- Automated tests

The API’s risk assessment and the experimental Transformer training pipeline are distinct parts of the project. The saved training metrics do not by themselves mean the Transformer is deployed as the API’s risk decision model.

## Data

- `data/subjective/` contains extracted SoccerMon CSV/JSON records for training load, wellness, injuries, illnesses, and game performance.
- `data/derived/` contains four per-athlete/per-session GPS summary CSVs for Team A and Team B in 2020 and 2021.
- `app/data/` contains the synthetic demo roster used by the web app: 14 football athletes with 30 days of records each.
- The large objective GPS archives are not included. To regenerate GPS summaries, download and process one archive at a time; keep raw archives outside the repository where possible.

The dataset is from [SoccerMon on Zenodo](https://doi.org/10.5281/zenodo.10033832) and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Run locally

The commands below use PowerShell on Windows. The interactive ATHLETEIQ demo uses the football sample data and needs both the backend and frontend running.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app\data\generate_demo_data.py
```

In the first terminal, start the API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.athleteiq_api:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the web app:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), then choose **Try your data** in the sidebar. Select a football athlete, edit any of the last 14 days of model inputs, and click **Run prediction**. The model checkpoint must be present under `artifacts/runs/20261104_153047_Transformer/` for predictions to run. The current checkpoint is missing its training-time scaling settings, so the app labels its result as illustrative. The trainer now saves those settings for future checkpoints.

To regenerate the fictional football sample roster and its records, run `python app\data\generate_demo_data.py` from the repository root.

## Research pipeline

Run the Transformer trainer after `data/pipeline.py` has generated `data/processed/daily_observations.csv`:

```powershell
python app/ml/train.py
```

For example, to train with BCE instead of the default Focal Loss:

```powershell
python app/ml/train.py --loss bce
```

Each training run writes `best_model.pt`, `train_log.csv`, and `results.json` under `artifacts/runs/<run-id>/`. The results include ROC-AUC, PR-AUC, threshold-based classification metrics and confusion counts, calibration metrics, and top-k risk precision, recall, and lift. Classification metrics use a default threshold of `0.5`; change it with `--decision-threshold`. Top-k metrics use `--top-k-pct`.

Run the tests with:

```powershell
pytest
```

## Main API routes

- `GET /health`
- `GET, POST /athletes`
- `GET, POST /observations`
- `GET, POST /injuries`
- `GET /assessments/{athlete_id}`

## Other sports

The demo data and trained model are football-only. The shared observation model can carry sport-specific values, but using this model for another sport would require appropriate training data and evaluation first.
