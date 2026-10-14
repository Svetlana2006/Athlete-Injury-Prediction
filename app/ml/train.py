import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["session_rpe", "duration_minutes", "total_distance_m", "high_speed_distance_m", "sleep_hours", "soreness", "fatigue", "wellness_score"]


def label_future_injury(frame: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    frame = frame.copy()
    frame["observed_on"] = pd.to_datetime(frame["observed_on"])
    frame["injury_on"] = pd.to_datetime(frame.get("injury_on"))
    frame["target"] = 0
    for athlete_id, group in frame.groupby("athlete_id"):
        injury_dates = group["injury_on"].dropna().tolist()
        for index, row in group.iterrows():
            frame.loc[index, "target"] = int(any(0 < (injury - row.observed_on).days <= horizon_days for injury in injury_dates))
    return frame


def main(input_dir: str = "data/processed", output: str = "artifacts/injury_risk_model.joblib") -> None:
    path = Path(input_dir) / "daily_observations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}. Run the Colab audit/normalisation notebook first.")
    data = pd.read_csv(path)
    missing = set(["athlete_id", "observed_on"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for feature in FEATURES:
        if feature not in data:
            data[feature] = pd.NA
    data = label_future_injury(data).sort_values("observed_on")
    split = max(1, int(len(data) * 0.8))
    train, test = data.iloc[:split], data.iloc[split:]
    if train.target.nunique() < 2:
        raise ValueError("Training interval has fewer than two target classes; choose a larger data period or inspect injury labels.")
    preprocess = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FEATURES)])
    model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000))])
    model.fit(train[FEATURES], train.target)
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    print("Time-aware holdout PR-AUC:", round(average_precision_score(test.target, probabilities), 4))
    print(classification_report(test.target, (probabilities >= 0.5).astype(int), zero_division=0))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output", default="artifacts/injury_risk_model.joblib")
    args = parser.parse_args()
    main(args.input_dir, args.output)
