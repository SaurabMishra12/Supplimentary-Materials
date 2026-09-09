"""Rebuild and verify all manuscript tables from raw records."""
import os
import pandas as pd

def reproduce_all_tables(data_dir: str, output_dir: str):
    print("Rebuilding tables from records...")
    # Table 2, 3, 4, 14, 20, 21, 25
