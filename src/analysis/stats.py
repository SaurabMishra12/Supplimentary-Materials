"""Statistical hypothesis testing and exposure-compliance decomposition."""
import numpy as np
from scipy import stats

def mcnemar_test(b: int, c: int) -> float:
    """Two-sided McNemar exact test with continuity correction."""
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial test on discordant pairs
    p_val = stats.binomtest(b, n, p=0.5, alternative="two-sided").pvalue
    return float(p_val)
