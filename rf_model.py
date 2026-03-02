"""Random Forest baseline model for molecular property prediction on Delaney."""

from pathlib import Path
import json

import joblib
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import random_split


RANDOM_SEED = 42
DATA_PATH = "data/delaney-processed.csv"
TARGET_COL = "measured log solubility in mols per litre"
DROP_COLS = [
	"Compound ID",
	"smiles",
	"ESOL predicted log solubility in mols per litre",
	TARGET_COL,
]
OUT_DIR = Path("models/RF_Delaney")


def load_features_and_target(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(data_path)
	feature_cols = [col for col in df.columns if col not in DROP_COLS]
	x = df[feature_cols]
	y = df[TARGET_COL]
	return x, y


def split_data(x: pd.DataFrame, y: pd.Series):
	"""Match GINE split exactly: torch random_split with 80/10/10 and fixed seed."""
	n_samples = len(x)
	train_size = int(0.8 * n_samples)
	val_size = int(0.1 * n_samples)
	test_size = n_samples - train_size - val_size

	dummy_dataset = torch.arange(n_samples)
	generator = torch.Generator().manual_seed(RANDOM_SEED)
	train_subset, val_subset, test_subset = random_split(
		dummy_dataset,
		[train_size, val_size, test_size],
		generator=generator,
	)

	x_train = x.iloc[train_subset.indices]
	x_val = x.iloc[val_subset.indices]
	x_test = x.iloc[test_subset.indices]
	y_train = y.iloc[train_subset.indices]
	y_val = y.iloc[val_subset.indices]
	y_test = y.iloc[test_subset.indices]
	return x_train, x_val, x_test, y_train, y_val, y_test


def evaluate(model: RandomForestRegressor, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
	preds = model.predict(x)
	return {
		"mse": float(mean_squared_error(y, preds)),
		"r2": float(r2_score(y, preds)),
	}


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	x, y = load_features_and_target(DATA_PATH)
	x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

	model = RandomForestRegressor(random_state=RANDOM_SEED)
	model.fit(x_train, y_train)

	train_metrics = evaluate(model, x_train, y_train)
	val_metrics = evaluate(model, x_val, y_val)
	test_metrics = evaluate(model, x_test, y_test)

	artifact = {
		"model": "RandomForestRegressor",
		"seed": RANDOM_SEED,
		"data_path": DATA_PATH,
		"target": TARGET_COL,
		"feature_columns": list(x.columns),
		"splits": {
			"train": int(len(x_train)),
			"val": int(len(x_val)),
			"test": int(len(x_test)),
		},
		"metrics": {
			"train": train_metrics,
			"val": val_metrics,
			"test": test_metrics,
		},
	}

	model_path = OUT_DIR / "rf_model.joblib"
	metrics_path = OUT_DIR / "metrics.json"

	joblib.dump(model, model_path)
	metrics_path.write_text(json.dumps(artifact, indent=2))

	print("Random Forest baseline complete")
	print(f"Saved model to: {model_path}")
	print(f"Saved metrics to: {metrics_path}")
	print(
		"Validation -> "
		f"MSE: {val_metrics['mse']:.6f}, R2: {val_metrics['r2']:.6f}"
	)
	print(f"Test -> MSE: {test_metrics['mse']:.6f}, R2: {test_metrics['r2']:.6f}")


if __name__ == "__main__":
	main()

