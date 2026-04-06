import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. DATA PROFILING
# -----------------------------
def profile_data(df, output_dir):
    """Save dataset profile: shape, dtypes, missing values."""
    os.makedirs(output_dir, exist_ok=True)

    profile_path = os.path.join(output_dir, "data_profile.txt")

    with open(profile_path, "w") as f:
        f.write("=== DATA PROFILE ===\n\n")

        # Shape
        f.write(f"Shape: {df.shape}\n\n")

        # Data types
        f.write("Data Types:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")

        # Missing values
        f.write("Missing Values:\n")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100

        for col in df.columns:
            f.write(f"{col}: {missing[col]} ({missing_pct[col]:.2f}%)\n")


# -----------------------------
# 2. DISTRIBUTION PLOTS
# -----------------------------
def plot_numeric_distributions(df, output_dir, numeric_cols):
    """Create histogram + KDE for numeric columns."""
    os.makedirs(output_dir, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

        filename = f"{col}_distribution.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


# -----------------------------
# 3. CORRELATION
# -----------------------------
def plot_correlation(df, output_dir, numeric_cols):
    """Generate correlation heatmap."""
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    return corr


# -----------------------------
# 4. MISSING DATA VISUALIZATION
# -----------------------------
def plot_missing(df, output_dir):
    """Visualize missing data."""
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False)

    plt.title("Missing Data Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_data.png"))
    plt.close()


# -----------------------------
# 5. OUTLIER DETECTION
# -----------------------------
def detect_outliers(df, numeric_cols, output_dir):
    """Detect outliers using IQR and save summary."""
    summary = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

        summary[col] = len(outliers)

    # Save to file
    path = os.path.join(output_dir, "outliers_summary.txt")
    with open(path, "w") as f:
        f.write("=== OUTLIER SUMMARY (IQR METHOD) ===\n\n")
        for col, count in summary.items():
            f.write(f"{col}: {count} outliers\n")

    return summary


# -----------------------------
# 6. MAIN FUNCTION
# -----------------------------
def generate_eda_report(df, config=None):
    """
    Generate a full automated EDA report.

    Parameters:
        df (pd.DataFrame)
        config (dict):
            - output_dir (str)
            - numeric_cols (list)
            - plot_style (str)
    """

    if config is None:
        config = {}

    output_dir = config.get("output_dir", "output")
    plot_style = config.get("plot_style", "whitegrid")
    numeric_cols = config.get("numeric_cols")

    # Apply style
    sns.set_style(plot_style)

    # Auto-detect numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    os.makedirs(output_dir, exist_ok=True)

    print("Running EDA report...")

    # Run all steps
    profile_data(df, output_dir)
    plot_numeric_distributions(df, output_dir, numeric_cols)
    plot_missing(df, output_dir)
    corr = plot_correlation(df, output_dir, numeric_cols)
    outliers = detect_outliers(df, numeric_cols, output_dir)

    print(f"EDA report saved in: {output_dir}")

    return {
        "correlation_matrix": corr,
        "outliers": outliers,
        "numeric_columns_used": numeric_cols,
    }
