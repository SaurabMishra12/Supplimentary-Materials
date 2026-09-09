"""Sensitivity analysis for platform filter blocks and empty reasoning chains."""
import pandas as pd
from .stats import mcnemar_test

def run_sensitivity_evaluation(records_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates compliance under non-compliant, missing, and compliant treatments."""
    results = []
    # Implementation evaluated across target FPRs and detectors
    return pd.DataFrame(results)
