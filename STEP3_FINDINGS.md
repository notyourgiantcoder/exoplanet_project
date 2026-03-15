# Exoplanet Dataset Cleaning - Step 3 Findings

## Artifacts Produced
- `imputation_distributions.png` (before vs after imputation distributions)
- `missing_values_bar.png` (missingness percentage by feature)
- `missing_heatmap.png` (row-level missingness map)
- `exoplanets_step3_clean.csv` (final cleaned table)

## Final Data Shape
- 6128 planets
- 15 columns total
- Structure: 2 identifier columns (`pl_name`, `hostname`) + 13 numeric modeling features

## Imputation Methodology
- Median imputation for low-missingness features (<10%).
- MICE-based imputation (`IterativeImputer`) with `RandomForestRegressor` for higher-missingness features:
  - `pl_orbeccen` (~15%)
  - `pl_eqt` (~25%)
  - `pl_insol` (~30%)
  - `st_age` (~21%)
- Log transforms applied to strongly right-skewed variables before imputation, then reversed to original units.

## Final Summary Statistics (from numeric features)
- `pl_rade`: mean 5.811, median 2.840, min 0.310, max 87.206
- `pl_bmasse`: mean 403.548, median 9.220, min 0.020, max 9534.852
- `pl_dens`: mean 4.861, median 2.560, min 0.005, max 2000.000
- `pl_orbeccen`: mean 0.090, median 0.000, min 0.000, max 0.950
- `pl_eqt`: mean 836.965, median 728.000, min 34.000, max 4050.000
- `pl_insol`: mean 341.295, median 59.115, min 0.000, max 44900.000
- `pl_orbper`: mean 6.984e4, median 11.122, min 0.091, max 4.020e8
- `pl_orbsmax`: mean 14.963, median 0.102, min 0.004, max 19000.000
- `st_teff`: mean 5406.221, median 5549.000, min 415.000, max 57000.000
- `st_lum`: mean -0.139, median -0.076, min -6.090, max 3.800
- `st_rad`: mean 1.477, median 0.951, min 0.012, max 88.475
- `st_mass`: mean 0.937, median 0.940, min 0.009, max 10.940
- `st_age`: mean 4.538, median 4.200, min 0.000, max 16.100

## Interpretation and Quality Notes
- No missing values remain in the selected modeling features after Step 3.
- The feature distributions indicate substantial right tails and possible outliers in several variables (`pl_orbper`, `pl_bmasse`, `pl_dens`, `pl_insol`, `pl_orbsmax`).
- For downstream modeling, prefer robust preprocessing (for example, log transforms and robust scaling) and outlier-resistant algorithms.
- Values in MICE-imputed columns are statistically inferred, so uncertainty should be considered during interpretation.
