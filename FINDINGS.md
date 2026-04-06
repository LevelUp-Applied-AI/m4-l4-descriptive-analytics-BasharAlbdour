# FINDINGS — Student Performance EDA

## Dataset Overview
- Shape: (2000, 10)
- Missing values handled:
  - commute_minutes → median imputation (~9%)
  - scholarship → filled with 'None' (~19%)

## Distribution Analysis
- GPA is slightly left-skewed.
- Study hours are approximately normally distributed.
- Attendance is centered around 70–90%.

## Correlation Analysis
- study_hours_weekly vs gpa: correlation = 0.64
- gpa vs attendance_pct: correlation = 0.04

## Hypothesis Testing
### Internship vs GPA
- Mean GPA (Internship): 2.983
- Mean GPA (No Internship): 2.701
- t = 13.564, p = 0.00000 (one-tailed), d = 0.706
- Note: A one-tailed p-value is reported because the hypothesis is directional — we predicted that students with internships would have *higher* GPA, not merely *different* GPA. scipy.stats.ttest_ind returns a two-tailed p-value by default, so the reported p-value is that result divided by 2. This is valid because the t-statistic is positive, confirming the observed difference is in the predicted direction.
- Result: Statistically significant difference.

### Scholarship vs Department
- chi2 = 17.136, p = 0.37686, dof = 16
- Result: No significant association.

### GPA across Departments (ANOVA)
- F-statistic = 0.6671, p = 0.614811
- Result: No significant difference between departments.

## Recommendations
- Encourage students to increase study hours.
- Expand internship opportunities.
- Investigate additional factors affecting GPA.
