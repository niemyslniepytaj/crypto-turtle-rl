import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataQualityService:
    """Ocena jakości wejściowych szeregów OHLCV przed symulacją."""

    EXCLUDE_COLUMNS = ["timestamp", "date", "symbol"]

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.getcwd()

    def _process_columns(self, df: pd.DataFrame):
        """Konwertuje kolumny typu object na datetime lub kody numeryczne."""
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[col + "_year"] = df[col].dt.year
                    df[col + "_month"] = df[col].dt.month
                    df[col + "_day"] = df[col].dt.day
                    df[col + "_hour"] = df[col].dt.hour
                    df[col + "_minute"] = df[col].dt.minute
                    df[col + "_second"] = df[col].dt.second
                    df.drop(col, axis=1, inplace=True)
                except Exception:
                    df[col] = df[col].astype('category').cat.codes
        return df

    def evaluate_metrics(self, df: pd.DataFrame):
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in self.EXCLUDE_COLUMNS]
        if len(numeric_cols) == 0:
            return None, {}

        numeric = df[numeric_cols]

        missing_ratio = numeric.isna().mean().mean()
        duplicate_ratio = df.duplicated().mean()
        avg_corr = numeric.corr().abs().stack().mean()
        std_mean = numeric.std().mean()

        # Syntetyczny Data Quality Score
        score = (
            (1 - missing_ratio) * 0.4 +
            (1 - duplicate_ratio) * 0.2 +
            (1 - abs(avg_corr - 0.5)) * 0.2 +
            (1 - np.tanh(std_mean / 10)) * 0.2
        ) * 100

        metrics = {
            "missing_ratio": round(missing_ratio, 5),
            "duplicate_ratio": round(duplicate_ratio, 5),
            "avg_corr": round(avg_corr, 5),
            "std_mean": round(std_mean, 5)
        }

        return round(score, 2), metrics, numeric_cols

    def calculate_quality(self, rabbit_msg_id=None, data_dir=None, data_file=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            data_path = os.path.join(self.base_dir, "shared", "files", data_dir, data_file)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Plik nie istnieje: {data_path}")

            df = pd.read_csv(data_path)
            df = self._process_columns(df)

            match = re.search(r"_set(\d+)_", data_file)
            set_name = f"set{match.group(1)}" if match else "unknown"

            score, metrics, numeric_features = self.evaluate_metrics(df)

            result_data = {
                "result": "success",
                "score": score,
                "metrics": metrics,
                "features": numeric_features,
                "set": set_name,
                "data_file": data_file,
                "timestamp": timestamp,
                "rabbit_msg_id": rabbit_msg_id
            }

            acc_dir = os.path.join(self.base_dir, "shared", "quality_reports", timestamp)
            os.makedirs(acc_dir, exist_ok=True)
            result_path = os.path.join(acc_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=4)

            log_path = os.path.join(acc_dir, "log.txt")
            with open(log_path, "w") as f:
                f.write(f"[{timestamp}] Data Quality calculation log\n")
                f.write(f"Data file: {data_file}\n")
                f.write(f"Set: {set_name}\n")
                f.write(f"Score: {score}\n")
                f.write(f"Metrics: {metrics}\n")
                f.write("Status: success\n")

            return result_data

        except Exception as e:
            print(f"Error in calculate_quality: {e}")
            return {
                "result": "failed",
                "message": str(e),
                "rabbit_msg_id": rabbit_msg_id,
                "data_file": data_file,
                "timestamp": timestamp
            }
