# Exoplanet Project Implementation (End-to-End, Detailed)

## 1) Project Objective

The goal of this project is to build an end-to-end machine learning pipeline that:

1. Cleans and engineers astrophysical exoplanet data.
2. Constructs a domain-informed target called Life Sustainability Score (LSS) on a 0 to 1 scale.
3. Trains multiple regression models to predict LSS.
4. Compares model performance and builds an ensemble.
5. Explains model behavior using SHAP global and local interpretability.

Primary source notebook used for implementation:
- exoplanet_preprocessing.ipynb

Primary model/evaluation outputs:
- outputs/model_comparison.csv
- outputs/model_comparison_with_ensemble.csv
- outputs/step9_model_comparison.png
- outputs/step9_actual_vs_predicted.png
- outputs/step10_leaderboard.png
- outputs/step11a_shap_global_importance.png
- outputs/step11b_shap_beeswarm.png
- outputs/step11c_shap_waterfall_kepler442b.png

---

## 2) Data Lineage and Artifact Flow

The implementation follows a strict staged artifact flow.

1. Raw source
- data/PSCompPars_2026.03.05_02.02.54.csv

2. Backup and selection
- exoplanets_raw_backup.csv
- exoplanets_selected.csv

3. Cleaning and feature conditioning
- exoplanets_step3_clean.csv
- exoplanets_step4_clean.csv
- exoplanets_step5_transformed.csv
- exoplanets_step6_final_features.csv
- metadata/transform_metadata.json (also mirrored at project root as transform_metadata.json in notebook flow)

4. Target creation outputs
- exoplanets_step7_model_ready.csv
- exoplanets_step7_with_lss.csv

5. Split and scaling outputs
- X_train.csv, X_test.csv, y_train.csv, y_test.csv
- X_train_scaled.csv, X_test_scaled.csv
- robust_scaler.pkl

6. Model artifacts
- model_ridge_regression.pkl
- model_random_forest.pkl
- model_xgboost.pkl
- model_neural_network.pkl
- model_ensemble.pkl

7. Evaluation and explainability outputs
- outputs/model_comparison.csv
- outputs/model_comparison_with_ensemble.csv
- outputs/*.png in Step 9, 10, 11

---

## 3) Full Pipeline Implementation Details

## Step 1: Load Raw Data

Implementation summary:
- Reads CSV with comment filtering to ignore commented metadata rows.
- Verifies load shape.
- Saves an immutable backup snapshot for reproducibility.

Code behavior:
- Input path: data/PSCompPars_2026.03.05_02.02.54.csv
- Loader: pandas read_csv with comment="#"
- Backup: exoplanets_raw_backup.csv

Observed initial shape in notebook logs:
- 6128 planets x 54 columns

Why this matters:
- Freezes the raw state before transformations.
- Enables reruns and auditability.

## Step 2: Select Core Columns

Selected schema:
- Identifier columns (2):
  - pl_name
  - hostname
- Planetary columns (6):
  - pl_rade, pl_bmasse, pl_dens, pl_orbeccen, pl_eqt, pl_insol
- Orbital columns (2):
  - pl_orbper, pl_orbsmax
- Stellar columns (5):
  - st_teff, st_lum, st_rad, st_mass, st_age

Total retained columns:
- 15 columns (2 IDs + 13 numeric modeling features)

Method decisions:
- Missingness report is generated per column.
- Columns are tagged for median vs MICE imputation strategy in advance.

Output artifact:
- exoplanets_selected.csv

Why this matters:
- Controls feature scope early.
- Keeps astrophysical interpretability while reducing noise from unused columns.

## Step 3: Missing Value Imputation

Input:
- exoplanets_selected.csv

Methodology used:

1. Hybrid imputation strategy
- Median imputation for relatively stable/low-missing columns.
- MICE-style iterative imputation for higher-missing and interdependent columns.

2. Imputer models
- SimpleImputer(strategy="median")
- IterativeImputer with estimator RandomForestRegressor(n_estimators=10, random_state=42), max_iter=10

3. Skew-aware preprocessing before imputation
- Right-skewed columns are temporarily log-transformed before imputation:
  - pl_orbper, pl_orbsmax, pl_bmasse, pl_dens, pl_insol, st_rad, st_mass
- Transform is reversed after imputation using expm1.

4. High-missing columns handled via iterative model
- pl_orbeccen, pl_eqt, pl_insol, st_age

Output:
- exoplanets_step3_clean.csv

Quality outcomes:
- Final missing values in selected numeric features: 0
- Dataset shape remains 6128 x 15

Interpretation notes:
- Random-forest-based iterative imputation captures nonlinear relationships.
- Notebook shows IterativeImputer convergence warning (early stopping criterion not reached). This is common and does not imply failure, but should be noted as a modeling uncertainty source.

## Step 4: Outlier Filtering

Input:
- exoplanets_step3_clean.csv

Method used:

1. Domain-rule filtering first
- Hard physical plausibility bounds applied feature-wise.
- Example bounds from implementation:
  - pl_bmasse: [0.1, 4131.0]
  - pl_rade: [0.3, 25.0]
  - pl_dens: [0.001, 100.0]
  - pl_orbper: [0.1, 100000]
  - pl_orbsmax: [0.001, 100.0]
  - pl_eqt: [50, 4000.0]
  - pl_insol: [0.0, 30000.0]
  - pl_orbeccen: [0.0, 0.95]
  - st_teff: [2000, 40000.0]
  - st_rad: [0.01, 50.0]
  - st_mass: [0.05, 8.0]
  - st_age: [0.0, 14.0]
  - st_lum: [-6.0, 3.8]

2. Statistical filtering second
- Z-score outlier removal with threshold |z| > 3.5.
- For heavy-tailed variables, z-scores are computed in log-space.

Shape effect from notebook logs:
- After domain cuts: 5894 rows
- After z-score cuts: 5456 rows

Output:
- exoplanets_step4_clean.csv

Why sequence matters:
- Domain cuts remove physically impossible extremes.
- Statistical cuts then reduce residual distributional anomalies.

## Step 5: Log Transform Selection

Input:
- exoplanets_step4_clean.csv

Method:
- Per-feature skewness before and after log1p is measured.
- A feature is transformed only if absolute skewness improvement >= 1.0.

Rules:
- st_lum is already logarithmic in source domain and is skipped.

Output artifacts:
- exoplanets_step5_transformed.csv
- transform_metadata.json

Recorded transformation decisions (from metadata):
- True (log-applied): pl_bmasse, pl_dens, pl_insol, pl_orbper, st_rad
- False (kept): pl_rade, pl_orbeccen, pl_eqt, pl_orbsmax, st_teff, st_lum, st_mass, st_age

Why this matters:
- Prevents over-transforming features that do not benefit.
- Creates reproducible transform trace via metadata file.

## Step 6: Collinearity Reduction (VIF + Correlation)

Input:
- exoplanets_step5_transformed.csv

Additional implementation detail:
- st_teff and st_mass are restored from step4 scale for interpretability.

Method stack:

1. Correlation scan
- Pairwise correlations flagged when |r| >= 0.5.

2. VIF computation
- Uses statsmodels variance_inflation_factor with constant term.
- Status thresholds:
  - VIF > 10: DROP
  - 5 < VIF <= 10: MONITOR
  - VIF <= 5: OK

3. Explicit drop list
- Dropped features:
  - pl_bmasse
  - pl_insol
  - pl_orbper
  - st_lum
  - st_rad

4. Kept modeling features (8)
- pl_rade
- pl_dens
- pl_orbeccen
- pl_eqt
- pl_orbsmax
- st_teff
- st_mass
- st_age

Output:
- exoplanets_step6_final_features.csv

Shape at this stage:
- 5456 x 10 (2 IDs + 8 model features)

Why this matters:
- Improves numerical stability.
- Reduces redundant signal that can inflate variance and hurt generalization.

## Step 7: LSS Target Construction

Input:
- Feature table: exoplanets_step6_final_features.csv
- Raw-scale support table: exoplanets_step4_clean.csv

Alignment step:
- Rows aligned by pl_name and sorted to guarantee one-to-one target attachment.

### 7.1 Component Scores

Five domain components are computed in [0, 1].

1. Habitable-zone proximity score
- Based on stellar luminosity-derived HZ inner/outer radii:
  - hz_inner = 0.95 * sqrt(L)
  - hz_outer = 1.67 * sqrt(L)
- Uses Gaussian decay from HZ center.

2. Temperature suitability score
- Uses equilibrium temperature around 255 K target.
- Combines a narrow optimal Gaussian and a broader partial-habitability Gaussian.

3. Atmospheric retention proxy
- Combines planet radius and density (Earth-like references).
- Adds a size gate penalty for very large planets.

4. Orbital stability score
- orbit_score = exp(-2 * eccentricity)

5. Stellar environment score
- Weighted mix of age, stellar temperature, and stellar mass suitability.

### 7.2 Weighted LSS Formula

Weights used in implementation:
- W_HZ = 0.35
- W_TEMP = 0.25
- W_RETENTION = 0.20
- W_ORBIT = 0.10
- W_STELLAR = 0.10

Final score:

LSS = clip(
  0.35 * hz_score
+ 0.25 * temp_score
+ 0.20 * retention_score
+ 0.10 * orbit_score
+ 0.10 * stellar_score,
0, 1)

Distribution diagnostics from notebook output:
- LSS mean: 0.3785
- LSS std: 0.1228
- Skewness: 0.896
- IQR: 0.1457
- Earth benchmark LSS: 0.9340

Output artifacts:
- exoplanets_step7_model_ready.csv (features + LSS)
- exoplanets_step7_with_lss.csv (features + component subscores + LSS)

Shape after step 7:
- 5456 x 11

## Step 8: Split and Robust Scaling

Input:
- exoplanets_step7_model_ready.csv

Feature set used for ML:
- pl_rade, pl_dens, pl_orbeccen, pl_eqt, pl_orbsmax, st_teff, st_mass, st_age

Target:
- LSS

Method:

1. Stratified regression split
- y is binned into 5 quantile bins using qcut.
- train_test_split with test_size=0.20, random_state=42, stratify=lss_bins.

2. Robust scaling
- RobustScaler fit only on training data.
- Transform applied to train and test.

Output artifacts:
- X_train.csv, X_test.csv, y_train.csv, y_test.csv
- X_train_scaled.csv, X_test_scaled.csv
- robust_scaler.pkl

Split sizes inferred from dataset size 5456:
- Train: 4364
- Test: 1092

Why RobustScaler:
- More stable than standard scaling under residual heavy tails and surviving outliers.

## Step 9: Train and Evaluate Base Models

Training input:
- TRAIN_TEST/X_train_scaled.csv
- TRAIN_TEST/X_test_scaled.csv
- TRAIN_TEST/y_train.csv
- TRAIN_TEST/y_test.csv

Evaluation metrics:
- R² Score
- MAE
- RMSE

### 9.1 Model 1: Ridge Regression
- Implementation: Ridge(alpha=1.0)
- Role: linear baseline with L2 regularization
- Results:
  - R² = 0.5741
  - MAE = 0.0528
  - RMSE = 0.0811

### 9.2 Model 2: Random Forest Regressor
- Hyperparameters:
  - n_estimators=300
  - max_depth=10
  - min_samples_leaf=4
  - random_state=42
  - n_jobs=-1
- Results:
  - R² = 0.9020
  - MAE = 0.0201
  - RMSE = 0.0389

### 9.3 Model 3: XGBoost Regressor
- Hyperparameters:
  - n_estimators=300
  - learning_rate=0.05
  - max_depth=6
  - subsample=0.8
  - colsample_bytree=0.8
  - random_state=42
  - verbosity=0
- Results:
  - R² = 0.9500
  - MAE = 0.0134
  - RMSE = 0.0278

### 9.4 Model 4: Neural Network (MLPRegressor)
- Hyperparameters:
  - hidden_layer_sizes=(128, 64, 32)
  - activation="relu"
  - max_iter=500
  - early_stopping=True
  - validation_fraction=0.1
  - random_state=42
- Results:
  - R² = 0.9032
  - MAE = 0.0194
  - RMSE = 0.0387

Step 9 output files:
- outputs/model_comparison.csv
- outputs/step9_model_comparison.png
- outputs/step9_actual_vs_predicted.png
- Saved model files at project root (model_*.pkl)

Best single model:
- XGBoost (R² 0.9500)

## Step 10: Voting Ensemble

Method:
- VotingRegressor over three non-linear learners:
  - Random Forest
  - XGBoost
  - Neural Network
- Each estimator uses the same hyperparameters as Step 9.

Result:
- R² = 0.9406
- MAE = 0.0143
- RMSE = 0.0303

Output artifacts:
- model_ensemble.pkl
- outputs/model_comparison_with_ensemble.csv
- outputs/step10_leaderboard.png

Interpretation:
- Ensemble improves stability and remains highly accurate.
- In this run, it is slightly behind XGBoost on all three metrics.

## Step 11: SHAP Explainability

Explainer setup:
- Model used for explanation: Random Forest (tree-based exact SHAP support).
- SHAP tool: shap.TreeExplainer
- Data explained: test set (1092 samples)

Reason for using Random Forest in SHAP stage:
- Efficient exact tree explanations.
- Clear feature attribution without requiring model-agnostic approximations.

Artifacts:
- outputs/step11a_shap_global_importance.png
- outputs/step11b_shap_beeswarm.png
- outputs/step11c_shap_waterfall_kepler442b.png

Final summary reported by notebook:
- Best model: XGBoost
- Best metrics: R² 0.9500, MAE 0.0134, RMSE 0.0278
- Top SHAP feature: Equilibrium Temp (K)

---

## 4) Detailed Interpretation of Output PNGs

## A) outputs/step9_model_comparison.png

What it shows:
- Three panels side-by-side for R², MAE, RMSE across four models.

Key findings visible in figure:
- XGBoost dominates all metrics:
  - Highest R² (0.9500)
  - Lowest MAE (0.0134)
  - Lowest RMSE (0.0278)
- Random Forest and Neural Network are nearly tied around R² ~0.90.
- Ridge is much weaker, showing the target is strongly nonlinear in available features.

Implication:
- Nonlinear models capture the LSS mapping far better than linear baseline.

## B) outputs/step9_actual_vs_predicted.png

What it shows:
- Scatter of actual LSS (x-axis) vs predicted LSS (y-axis) for best model (XGBoost).
- Dashed red diagonal is perfect-prediction line.

Key visual findings:
- Most points cluster close to the diagonal, indicating high calibration.
- Error spread widens modestly at higher LSS values, common in upper-tail regression.
- A few visible outliers are present but sparse relative to total point cloud.

Implication:
- High overall fidelity with slight heteroscedastic behavior near top-score planets.

## C) outputs/step10_leaderboard.png

What it shows:
- Horizontal R² ranking including ensemble.

Ordered results in plot:
1. XGBoost: 0.9500
2. Voting Ensemble: 0.9406
3. Neural Network: 0.9032
4. Random Forest: 0.9020
5. Ridge Regression: 0.5741

Key insight:
- Ensemble is second-best and clearly above RF/NN singles, but still below XGBoost.

Implication:
- The XGBoost hypothesis class appears best aligned with this feature-target mapping.

## D) outputs/step11a_shap_global_importance.png

What it shows:
- Mean absolute SHAP value by feature (global importance) on Random Forest test predictions.

Reported bars (approx values shown on figure):
- Equilibrium Temp (K): 0.0660
- Planet Radius (R⊕): 0.0550
- Planet Density: 0.0097
- Semi-Major Axis (AU): 0.0070
- Orbital Eccentricity: 0.0057
- Stellar Age (Gyr): 0.0026
- Stellar Mass (M☉): 0.0023
- Stellar Temperature (K): 0.0011

Key insight:
- Temperature and radius dominate attribution magnitude.
- Stellar features contribute less in this trained model relative to planetary/orbital proxies.

Implication:
- Model decisions are primarily driven by thermodynamic and size-related suitability signals.

## E) outputs/step11b_shap_beeswarm.png

What it shows:
- Distribution of signed SHAP values per feature.
- Color encodes normalized raw feature value (red high, blue low).

Pattern-level interpretation:
- Equilibrium temperature has widest SHAP spread, confirming strongest non-linear effect range.
- Planet radius also has large effect spread.
- Lower-impact features cluster near 0 SHAP with tighter clouds.
- Directionality is feature-dependent and nonlinear (high value does not always imply higher LSS globally).

Implication:
- The model encodes nonlinear astrophysical interactions rather than simple monotonic rules.

## F) outputs/step11c_shap_waterfall_kepler442b.png

What it shows:
- Local explanation for Kepler-442 b prediction.
- Base value, additive SHAP contributions, predicted LSS, and actual LSS markers.

Values visible in figure:
- Base value: 0.3783
- Predicted LSS: 0.8299
- Actual LSS: 0.9691

Largest positive contributors:
- Equilibrium Temp (K): +0.2992
- Planet Radius (R⊕): +0.1217
- Smaller positive boosts from semi-major axis, density, eccentricity

Small negative contributors:
- Stellar age, stellar mass, stellar temperature (minor downward adjustments)

Interpretation:
- The model strongly increases this planet from baseline due mainly to thermal and radius suitability.
- Predicted value is high but underestimates the engineered target by about 0.1392, indicating a residual for this high-potential case.

---

## 5) Consolidated Quantitative Results

From outputs/model_comparison_with_ensemble.csv:

- Ridge Regression: R² 0.5741, MAE 0.0528, RMSE 0.0811
- Random Forest: R² 0.9020, MAE 0.0201, RMSE 0.0389
- XGBoost: R² 0.9500, MAE 0.0134, RMSE 0.0278
- Neural Network: R² 0.9032, MAE 0.0194, RMSE 0.0387
- Voting Ensemble: R² 0.9406, MAE 0.0143, RMSE 0.0303

Final winner:
- XGBoost

---

## 6) Implementation Strengths

1. Strong reproducibility
- Deterministic seeds and staged artifact persistence.

2. Hybrid data treatment
- Domain bounds + statistical outlier handling + skew-aware transforms.

3. Target engineering transparency
- LSS decomposed into interpretable component scores and explicit weights.

4. Broad model family coverage
- Linear, bagging, boosting, neural network, and ensemble comparisons.

5. Explainability included
- Global and local SHAP outputs for interpretability and trust.

---

## 7) Observed Limitations and Technical Notes

1. Iterative imputer warning
- Early stopping criterion not reached in Step 3 logs.
- Recommendation: sensitivity analysis on max_iter and estimator complexity.

2. Single-split evaluation
- Current metrics come from one 80/20 split.
- Recommendation: repeated CV or nested CV for robustness.

3. Mixed model/explainer pairing
- Best predictor is XGBoost, but SHAP plots are produced from Random Forest.
- This is valid but should be interpreted as explaining RF behavior, not XGBoost behavior.

4. High-LSS residuals
- Scatter and waterfall indicate slightly larger residuals near upper target range.
- Recommendation: quantile error analysis and calibration checks in upper decile.

---

## 8) Reproduction Checklist

1. Run steps 1 to 8 in exoplanet_preprocessing.ipynb to rebuild cleaned and split datasets.
2. Run step 9 to train base models and save model_comparison.csv and step9 plots.
3. Run step 10 to build voting ensemble and save leaderboard outputs.
4. Run step 11 to generate SHAP global, beeswarm, and Kepler-442 b waterfall plots.
5. Verify outputs directory contains all six PNG files and both CSV summaries.

---

## 9) Final Technical Conclusion

The implementation is a complete, production-style ML workflow for exoplanet habitability scoring. The strongest predictor is XGBoost, which achieves high explanatory fit (R² 0.9500) with low absolute and squared errors. Explainability outputs show the learned signal is dominated by equilibrium temperature and planet radius, matching domain intuition for habitability proxies. The project successfully combines domain-informed target engineering, robust preprocessing, strong predictive modeling, and interpretable diagnostics in a reproducible artifact chain.
