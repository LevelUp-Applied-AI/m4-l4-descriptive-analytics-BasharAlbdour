import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from eda_report import generate_eda_report


def create_sample_df():
    """Create a small sample dataframe for testing."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 100],  # has outlier
            "b": [10, 20, 30, 40, 50],
            "c": [5, np.nan, 7, 8, 9],
        }
    )


def test_generate_eda_report_runs(tmp_path):
    """Test that the function runs without errors."""
    df = create_sample_df()

    output_dir = tmp_path / "eda_output"

    result = generate_eda_report(df, {"output_dir": str(output_dir)})

    assert result is not None


def test_output_files_created(tmp_path):
    """Test that key output files are created."""
    df = create_sample_df()

    output_dir = tmp_path / "eda_output"

    generate_eda_report(df, {"output_dir": str(output_dir)})

    # Check important files
    assert os.path.exists(output_dir / "data_profile.txt")
    assert os.path.exists(output_dir / "correlation_heatmap.png")
    assert os.path.exists(output_dir / "missing_data.png")
    assert os.path.exists(output_dir / "outliers_summary.txt")


def test_outlier_detection(tmp_path):
    """Test that outliers are detected correctly."""
    df = create_sample_df()

    output_dir = tmp_path / "eda_output"

    result = generate_eda_report(df, {"output_dir": str(output_dir)})

    outliers = result["outliers"]

    # column 'a' has 100 as outlier
    assert outliers["a"] > 0


def test_numeric_column_detection(tmp_path):
    """Test automatic numeric column detection."""
    df = create_sample_df()

    output_dir = tmp_path / "eda_output"

    result = generate_eda_report(df, {"output_dir": str(output_dir)})

    numeric_cols = result["numeric_columns_used"]

    assert "a" in numeric_cols
    assert "b" in numeric_cols


def test_handles_missing_values(tmp_path):
    df = create_sample_df()

    output_dir = tmp_path / "eda_output"

    generate_eda_report(df, {"output_dir": str(output_dir)})

    assert True
