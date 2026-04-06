import pandas as pd
import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy import stats


def bootstrap_mean_ci(data, n_bootstrap=10000, ci=95):
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data (array-like)
        n_bootstrap (int)
        ci (int): confidence level

    Returns:
        (lower, upper)
    """

    data = np.array(data)
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)

    return float(lower), float(upper)


def bootstrap_gpa_by_internship(df):
    """
    Compute bootstrap CI for GPA by internship status.
    """

    gpa_yes = df[df["has_internship"] == "Yes"]["gpa"].dropna()
    gpa_no = df[df["has_internship"] == "No"]["gpa"].dropna()

    ci_yes = bootstrap_mean_ci(gpa_yes)
    ci_no = bootstrap_mean_ci(gpa_no)

    print("\nBootstrap 95% CI for GPA:")
    print(f"Internship: {ci_yes}")
    print(f"No Internship: {ci_no}")

    # Parametric CI for comparison
    n = len(gpa_yes)
    se = gpa_yes.std() / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    param_ci = (gpa_yes.mean() - t_crit * se, gpa_yes.mean() + t_crit * se)
    print(f"Parametric 95% CI (internship group): {param_ci}")

    return {"internship": ci_yes, "no_internship": ci_no}


def power_analysis_from_data(df):
    """
    Estimate required sample size to detect observed effect.
    """

    gpa_yes = df[df["has_internship"] == "Yes"]["gpa"].dropna()
    gpa_no = df[df["has_internship"] == "No"]["gpa"].dropna()

    # Effect size (Cohen's d)
    mean_diff = gpa_yes.mean() - gpa_no.mean()
    pooled_std = ((gpa_yes.std() ** 2 + gpa_no.std() ** 2) / 2) ** 0.5
    effect_size = mean_diff / pooled_std

    analysis = TTestIndPower()

    sample_size = analysis.solve_power(effect_size=effect_size, power=0.8, alpha=0.05)

    print("\nPower Analysis:")
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    print(f"Required sample size per group: {sample_size:.0f}")

    return sample_size


def simulate_false_positive_rate(n_simulations=1000, sample_size=50, alpha=0.05):
    """
    Simulate hypothesis tests where NO real difference exists.
    Measure how often we incorrectly reject the null hypothesis.
    """

    false_positives = 0

    for _ in range(n_simulations):
        # Two groups with SAME mean → no real difference
        group1 = np.random.normal(loc=3.0, scale=0.5, size=sample_size)
        group2 = np.random.normal(loc=3.0, scale=0.5, size=sample_size)

        t_stat, p_value = stats.ttest_ind(group1, group2)

        if p_value < alpha:
            false_positives += 1

    rate = false_positives / n_simulations

    print("\nSimulation Results:")
    print(f"False positive rate: {rate:.4f} (expected ≈ {alpha})")

    return rate
