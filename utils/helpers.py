import numpy as np
import pandas as pd
from datetime import datetime

def parse_age(age_str):
    """Parse age string to numeric value."""
    if "-" in age_str:
        age_parts = age_str.split("-")
        if len(age_parts) == 2:
            return (int(age_parts[0]) + int(age_parts[1])) / 2
    return int(age_str)

def clean_for_json(obj):
    """Clean data for JSON serialization by handling numpy types, NaN, etc."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, datetime.date):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj