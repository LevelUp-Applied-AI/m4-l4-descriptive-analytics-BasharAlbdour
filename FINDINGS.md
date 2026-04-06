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
- t = 13.564, p = 0.00000, d = 0.706
- Result: Statistically significant difference.

### Scholarship vs Department
- chi2 = 13.949, p = 0.30401, dof = 12
- Result: No significant association.

## Recommendations
- Encourage students to increase study hours.
- Expand internship opportunities.
- Investigate additional factors affecting GPA.
