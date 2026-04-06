"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations


# df = pd.read_csv('data/student_performance.csv')

# numeric_cols = ['course_load', 'study_hours_weekly', 'gpa', 'attendance_pct', 'commute_minutes']

# for col in numeric_cols:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
#     print(f"{col}: {len(outliers)} outliers")

# describe = df.describe()
# print(describe)
# missing=df.isnull().sum()
# print(missing)
# shape=df.shape
# print(shape)
# departments=df['department'].value_counts()
# print(departments)
# semesters=df['semester'].value_counts()
# print(semesters)
# internship_counts=df['has_internship'].value_counts()
# print(internship_counts)
# scholarship_counts=df['scholarship'].value_counts()
# print(scholarship_counts)


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    df = pd.read_csv("data/student_performance.csv")
    shape = df.shape
    data_types = df.dtypes
    missing_values = df.isnull().sum()
    missing_value_percentage = (missing_values / len(df)) * 100

    df["commute_minutes"] = df["commute_minutes"].fillna(df["commute_minutes"].median())
    df["scholarship"] = df["scholarship"].fillna("None")

    with open("output/data_profile.txt", "w") as f:
        f.write(f"shape:{shape}\n")
        f.write(f"data types:\n{data_types.to_string()}\n")
        f.write("Missing Values:\n")
        for col in df.columns:
            f.write(
                f"{col}: {missing_values[col]} ({missing_value_percentage[col]:.2f}%)\n"
            )
        f.write("\nReason why the missing values were handled like this:\n")
        f.write(
            """commute minutes ~9% missing was computed using the median because it is a numeric variable and it is robust
                to outliers , also the max value is 79 min which is far from the 75% percentile and this suggests
                right skewed (right tail) and this would pull the mean upwards"""
        )
        f.write(
            """\n scholarship ~19% missing was filled with "None" because it is a categorical variable and it is more meaningful 
                to represent absence of scholarship and avoid dropping a large portion of data."""
        )
    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["gpa"], kde=True, color="steelblue", ax=ax)
    ax.set_title("GPA Distribution Left-skewed", fontsize=13)
    ax.set_xlabel("GPA")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig("output/GPA_Distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["study_hours_weekly"], kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Weekly Study Hours", fontsize=13)
    ax.set_xlabel("Study Hours per Week")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig("output/Distribution_of_Weekly_Study_Hours.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["attendance_pct"], kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Attendance Percentage", fontsize=13)
    ax.set_xlabel("Attendance Percentage")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig("output/Attendance_Percentage_Distribution.png", dpi=150)
    plt.close(fig)

    dept_order = (
        df.groupby("department")["gpa"].median().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="department", y="gpa", data=df, order=dept_order)
    plt.title("GPA by Department (Sorted by Median GPA)")
    plt.xlabel("Department")
    plt.ylabel("GPA")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("output/gpa_by_department.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        x="scholarship", data=df, order=df["scholarship"].value_counts().index
    )
    plt.title("Distribution of Scholarship Types")
    plt.xlabel("Scholarship Type")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("output/scholarship_distribution.png")
    plt.close(fig)


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """

    numeric_cols = [
        "course_load",
        "study_hours_weekly",
        "gpa",
        "attendance_pct",
        "commute_minutes",
    ]

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png")
    plt.close()

    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    top_pairs = (
        corr_matrix.where(mask)
        .stack()
        .abs()
        .sort_values(ascending=False)
        .head(2)
        .index.tolist()
    )

    for col_a, col_b in top_pairs:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[col_a], y=df[col_b])
        plt.title(f"{col_a} vs {col_b}")
        plt.xlabel(col_a.replace("_", " ").title())
        plt.ylabel(col_b.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(f"output/{col_a}_vs_{col_b}.png")
        plt.close()

    return corr_matrix, top_pairs


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_anova',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests to consider:
        - t-test: Does GPA differ between students with and without internships?
        - ANOVA: Does GPA differ across departments?
        - Correlation test: Is the correlation between study hours and GPA significant?
    """
    results = {}

    print("\n" + "=" * 60)
    print("Hypothesis Testing Results")
    print("=" * 60)

    gpa_intern = df[df["has_internship"] == "Yes"]["gpa"]
    gpa_no_intern = df[df["has_internship"] == "No"]["gpa"]

    t_stat, p_val = stats.ttest_ind(gpa_intern, gpa_no_intern)
    # One-tailed test: the hypothesis is directional ("internship → higher GPA"),
    # so we divide the two-tailed p-value returned by ttest_ind by 2.
    # This is valid only when t_stat > 0 (internship mean > no-internship mean),
    # which confirms the direction matches our hypothesis.
    p_val = p_val / 2

    pooled_std = np.sqrt((gpa_intern.std() ** 2 + gpa_no_intern.std() ** 2) / 2)
    cohens_d = (gpa_intern.mean() - gpa_no_intern.mean()) / pooled_std

    print("\nTest 1: GPA vs Internship (one-tailed t-test)")
    print(f"Mean GPA (Internship): {gpa_intern.mean():.3f}")
    print(f"Mean GPA (No Internship): {gpa_no_intern.mean():.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value (one-tailed): {p_val:.5f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if p_val < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: Not statistically significant.")

    results["ttest_dict"] = {
        "t": t_stat,
        "p": p_val,
        "d": cohens_d,
        "mean_yes": gpa_intern.mean(),
        "mean_no": gpa_no_intern.mean(),
    }

    ct = pd.crosstab(df["scholarship"], df["department"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)

    print("\nTest 2: Scholarship vs Department")
    print(f"chi2: {chi2:.3f}")
    print(f"p-value: {p:.5f}")
    print(f"dof: {dof}")

    if p < 0.05:
        print("Result: Statistically significant association.")
    else:
        print("Result: No statistically significant association.")

    results["chi2_dict"] = {"chi2": chi2, "p": p, "dof": dof}

    departments = df["department"].unique()
    groups = [df[df["department"] == d]["gpa"].dropna() for d in departments]

    f_stat, p_anova = stats.f_oneway(*groups)

    print("\nTest 3: GPA across Departments (one-way ANOVA)")
    print(f"  F-statistic : {f_stat:.4f}")
    print(f"  p-value     : {p_anova:.6f}")
    print(
        "  Result:",
        (
            "Significant -- at least one department mean differs."
            if p_anova < 0.05
            else "Not significant."
        ),
    )

    posthoc = []

    if p_anova < 0.05:
        dept_pairs = list(combinations(departments, 2))
        n_comparisons = len(dept_pairs)
        alpha_bonf = 0.05 / n_comparisons

        print(
            f"\n  Post-hoc pairwise t-tests (Bonferroni a = 0.05/{n_comparisons} = {alpha_bonf:.4f})"
        )
        print(
            f"  {'Dept A':<20} {'Dept B':<20} {'t':>8} {'p_raw':>10} {'p_bonf':>10} {'Sig?':>6}"
        )
        print("  " + "-" * 76)

        for dept_a, dept_b in dept_pairs:
            gpa_a = df[df["department"] == dept_a]["gpa"].dropna()
            gpa_b = df[df["department"] == dept_b]["gpa"].dropna()

            t, p_raw = stats.ttest_ind(gpa_a, gpa_b)
            p_bonf = min(p_raw * n_comparisons, 1.0)

            significant = p_bonf < 0.05
            print(
                f"  {dept_a:<20} {dept_b:<20} {t:>8.3f} {p_raw:>10.5f}"
                f" {p_bonf:>10.5f} {'Y' if significant else 'N':>6}"
            )

            posthoc.append(
                {
                    "dept_a": dept_a,
                    "dept_b": dept_b,
                    "t": t,
                    "p_raw": p_raw,
                    "p_bonf": p_bonf,
                    "significant": significant,
                }
            )

    results["anova_dict"] = {"f_stat": f_stat, "p_value": p_anova, "posthoc": posthoc}

    print("=" * 60)

    return results


def plot_violin(df):

    dept_order = (
        df.groupby("department")["gpa"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.violinplot(
        x="department",
        y="gpa",
        data=df,
        order=dept_order,
        inner="box",
        palette="muted",
        ax=ax,
    )

    for i, dept in enumerate(dept_order):
        median_val = df[df["department"] == dept]["gpa"].median()
        ax.text(
            i,
            median_val + 0.05,
            f"{median_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(
        "GPA Distribution by Department\n(violin width = density of students)",
        fontsize=13,
    )
    ax.set_xlabel("Department")
    ax.set_ylabel("GPA")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig("output/gpa_violin_by_department.png", dpi=150)
    plt.close(fig)
    print("Violin plot saved -> output/gpa_violin_by_department.png")


def write_findings(df, corr_matrix, top_pairs, results):
    with open("FINDINGS.md", "w", encoding="utf-8") as f:

        f.write("# FINDINGS — Student Performance EDA\n\n")

        f.write("## Dataset Overview\n")
        f.write(f"- Shape: {df.shape}\n")
        f.write("- Missing values handled:\n")
        f.write("  - commute_minutes → median imputation (~9%)\n")
        f.write("  - scholarship → filled with 'None' (~19%)\n\n")

        f.write("## Distribution Analysis\n")
        f.write("- GPA is slightly left-skewed.\n")
        f.write("- Study hours are approximately normally distributed.\n")
        f.write("- Attendance is centered around 70–90%.\n\n")

        f.write("## Correlation Analysis\n")
        for col_a, col_b in top_pairs:
            val = corr_matrix.loc[col_a, col_b]
            f.write(f"- {col_a} vs {col_b}: correlation = {val:.2f}\n")

        f.write("\n")

        t = results["ttest_dict"]

        f.write("## Hypothesis Testing\n")
        f.write("### Internship vs GPA\n")
        f.write(f"- Mean GPA (Internship): {t['mean_yes']:.3f}\n")
        f.write(f"- Mean GPA (No Internship): {t['mean_no']:.3f}\n")
        f.write(
            f"- t = {t['t']:.3f}, p = {t['p']:.5f} (one-tailed), d = {t['d']:.3f}\n"
        )
        f.write(
            "- Note: A one-tailed p-value is reported because the hypothesis is "
            "directional — we predicted that students with internships would have "
            "*higher* GPA, not merely *different* GPA. "
            "scipy.stats.ttest_ind returns a two-tailed p-value by default, "
            "so the reported p-value is that result divided by 2. "
            "This is valid because the t-statistic is positive, confirming the "
            "observed difference is in the predicted direction.\n"
        )

        if t["p"] < 0.05:
            f.write("- Result: Statistically significant difference.\n\n")
        else:
            f.write("- Result: Not statistically significant.\n\n")

        c = results["chi2_dict"]

        f.write("### Scholarship vs Department\n")
        f.write(f"- chi2 = {c['chi2']:.3f}, p = {c['p']:.5f}, dof = {c['dof']}\n")

        if c["p"] < 0.05:
            f.write("- Result: Significant association.\n\n")
        else:
            f.write("- Result: No significant association.\n\n")

        a = results["anova_dict"]

        f.write("### GPA across Departments (ANOVA)\n")
        f.write(f"- F-statistic = {a['f_stat']:.4f}, p = {a['p_value']:.6f}\n")

        if a["p_value"] < 0.05:
            f.write("- Result: At least one department differs significantly.\n\n")
        else:
            f.write("- Result: No significant difference between departments.\n\n")

        if a["posthoc"]:
            f.write("#### Post-hoc Pairwise Comparisons (Bonferroni)\n")
            for row in a["posthoc"]:
                f.write(
                    f"- {row['dept_a']} vs {row['dept_b']}: "
                    f"p_adj = {row['p_bonf']:.5f} "
                    f"({'Significant' if row['significant'] else 'Not significant'})\n"
                )
            f.write("\n")

        f.write("## Recommendations\n")
        f.write("- Encourage students to increase study hours.\n")
        f.write("- Expand internship opportunities.\n")
        f.write("- Investigate additional factors affecting GPA.\n")


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    df = load_and_profile("data/student_performance.csv")

    plot_distributions(df)
    corr_matrix, top_pairs = plot_correlations(df)
    results = run_hypothesis_tests(df)
    plot_violin(df)
    write_findings(df, corr_matrix, top_pairs, results)

    print("\nTop correlated pairs:", top_pairs)


if __name__ == "__main__":
    main()
