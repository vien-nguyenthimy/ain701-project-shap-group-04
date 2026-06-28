import pandas as pd
import sys
from pathlib import Path


def load_data(file_path: str = None) -> pd.DataFrame:
    if file_path is None:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / "data" / "raw" / "Loan_default.csv"
    
    return pd.read_csv(file_path)


def get_data_info(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "head": df.head().to_dict()
    }