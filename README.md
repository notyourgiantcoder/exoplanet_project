# 🌌 Beyond Binary: Predicting Life Sustainability Scores for Exoplanets

A machine-learning-oriented framework for evaluating **exoplanet life sustainability** using scientifically derived planetary characteristics.

Instead of treating habitability as a simple binary classification problem, this project introduces a continuous **Life Sustainability Score (LSS)** designed to represent how favorable an exoplanet's observed characteristics are for sustaining life.

---

## 🎯 Project Overview

Traditional approaches to exoplanet habitability often reduce the question to:

> **Habitable vs. Not Habitable**

This project explores a more granular approach by constructing a **continuous Life Sustainability Score (LSS)** from measurable exoplanet properties.

The current implementation focuses on building a **clean, statistically reliable, and machine-learning-ready dataset** through a structured preprocessing pipeline.

The resulting dataset can serve as the foundation for predictive modeling and explainable analysis in future stages.

---

## 🚀 Current Implementation

The current version of the project implements the complete **data preprocessing and LSS construction pipeline**.

### Implemented Components

* Raw exoplanet data ingestion
* Selection of scientifically relevant features
* Missing-value imputation
* Outlier detection and filtering
* Feature transformation
* Log transformation of skewed variables
* Multicollinearity analysis and reduction using VIF
* Construction of the continuous **Life Sustainability Score (LSS)**
* Train/test dataset generation
* Robust feature scaling
* Export of model-ready datasets
* Transformation metadata preservation

---

# 🔬 Methodology

The preprocessing pipeline follows a sequential workflow:

```text
Raw Exoplanet Dataset
          │
          ▼
   Feature Selection
          │
          ▼
 Missing Value Handling
          │
          ▼
    Outlier Filtering
          │
          ▼
   Feature Transformation
          │
          ▼
 Multicollinearity Reduction
        (VIF)
          │
          ▼
      LSS Creation
          │
          ▼
   Train / Test Split
          │
          ▼
    Robust Scaling
          │
          ▼
   ML-Ready Dataset
```

---

## 🪐 Life Sustainability Score (LSS)

The **Life Sustainability Score** is a continuous target constructed using scientifically motivated proxy indicators derived from observed exoplanet characteristics.

Rather than assigning an exoplanet to a simple `0/1` class, the LSS provides a continuous measure that can subsequently be used for:

* Ranking exoplanets
* Regression-based prediction
* Comparative analysis
* Feature importance analysis
* Model interpretability
* Future habitability research

The LSS dataset is generated during the preprocessing pipeline and stored for downstream machine learning experiments.

---

# 🧹 Data Preprocessing Pipeline

The preprocessing workflow is implemented in:

```text
exoplanet_preprocessing.ipynb
```

### 1. Raw Data Loading

The original exoplanet catalogue is loaded and examined to understand:

* Available features
* Missingness
* Feature distributions
* Data types
* Potentially relevant planetary characteristics

### 2. Feature Selection

Only the core variables required for the analysis are retained.

This reduces unnecessary dimensionality and establishes a consistent feature space for subsequent processing.

### 3. Missing Value Imputation

Missing observations are handled before statistical transformations and model preparation.

This prevents incomplete records from propagating through subsequent stages of the pipeline.

### 4. Outlier Filtering

Extreme observations are identified and filtered using appropriate statistical criteria.

This helps reduce the influence of anomalous measurements on downstream analysis.

### 5. Log Transformation

Highly skewed numerical variables are transformed where appropriate.

This helps:

* Reduce skewness
* Stabilize variance
* Improve feature distributions
* Make relationships more suitable for statistical and machine-learning models

### 6. Multicollinearity Reduction

Variance Inflation Factor (**VIF**) analysis is used to identify highly correlated predictors.

Redundant features are removed to produce a more stable feature set for future machine-learning models.

### 7. LSS Construction

The processed scientific indicators are combined to construct the continuous **Life Sustainability Score**.

The resulting scores are stored alongside the processed exoplanet data.

### 8. Train/Test Preparation

The processed data is separated into training and testing datasets.

A `RobustScaler` is then used to scale numerical features while reducing sensitivity to remaining extreme values.

---

# 📁 Project Structure

```text
Beyond-Binary-Predicting-Life-Sustainability-Scores-for-Exoplanets/
│
├── exoplanet_preprocessing.ipynb
│
├── PSCompPars_2026.03.05_02.02.54.csv
├── exoplanets_raw_backup.csv
│
├── exoplanets_selected.csv
├── exoplanets_step3_clean.csv
├── exoplanets_step4_clean.csv
├── exoplanets_step5_transformed.csv
├── exoplanets_step6_final_features.csv
│
├── exoplanets_step7_model_ready.csv
├── exoplanets_step7_with_lss.csv
│
├── transform_metadata.json
├── robust_scaler.pkl
│
├── X_train.csv
├── X_test.csv
├── X_train_scaled.csv
├── X_test_scaled.csv
│
├── y_train.csv
└── y_test.csv
```

---

# 📊 Dataset Files

| File                                  | Description                                 |
| ------------------------------------- | ------------------------------------------- |
| `PSCompPars_2026.03.05_02.02.54.csv`  | Original raw exoplanet catalogue            |
| `exoplanets_raw_backup.csv`           | Backup of the original dataset              |
| `exoplanets_selected.csv`             | Selected core features                      |
| `exoplanets_step3_clean.csv`          | Dataset after missing-value handling        |
| `exoplanets_step4_clean.csv`          | Dataset after outlier filtering             |
| `exoplanets_step5_transformed.csv`    | Dataset after feature transformations       |
| `exoplanets_step6_final_features.csv` | Final features after VIF reduction          |
| `exoplanets_step7_model_ready.csv`    | ML-ready dataset                            |
| `exoplanets_step7_with_lss.csv`       | Dataset containing calculated LSS values    |
| `transform_metadata.json`             | Metadata describing applied transformations |
| `robust_scaler.pkl`                   | Fitted RobustScaler                         |
| `X_train.csv`                         | Training features                           |
| `X_test.csv`                          | Testing features                            |
| `X_train_scaled.csv`                  | Scaled training features                    |
| `X_test_scaled.csv`                   | Scaled testing features                     |
| `y_train.csv`                         | Training LSS values                         |
| `y_test.csv`                          | Testing LSS values                          |

---

# 🔭 Future Scope

The current preprocessing and LSS construction pipeline provides the foundation for a broader machine-learning framework.

The planned next stages are:

### 01 — Develop an ML Framework for Exoplanet Sustainability Assessment

Develop and compare machine-learning models capable of learning the relationship between exoplanet characteristics and their corresponding Life Sustainability Scores.

Potential approaches include regression-based ensemble models and other suitable supervised learning techniques.

---

### 02 — Predict a Continuous Life Sustainability Score

Train predictive models to estimate the **continuous LSS** of exoplanets for which complete measurements may not be directly available.

This would transform the current LSS construction framework into a predictive system.

---

### 03 — Model Planetary, Orbital & Stellar Interactions

Extend the feature space beyond individual planetary characteristics by incorporating relationships between:

* Planetary properties
* Orbital parameters
* Host-star characteristics

This would allow the framework to account for the broader planetary system environment.

---

### 04 — Construct LSS Using Scientifically Derived Proxy Indicators

Further refine the LSS formulation using scientifically motivated proxy indicators associated with planetary and stellar conditions.

The objective is to make the score increasingly representative of the physical conditions relevant to life sustainability.

---

### 05 — Hyperparameter Optimization & Cross-Validation

Improve predictive performance through:

* Hyperparameter tuning
* K-Fold cross-validation
* Model comparison
* Performance benchmarking

This stage will help identify the most reliable model configuration while reducing the risk of overfitting.

---

### 06 — Retrieval-Augmented Generation for Scientific Evidence

Integrate a **Retrieval-Augmented Generation (RAG)** layer to retrieve relevant scientific literature and evidence supporting the indicators and relationships used by the framework.

This could provide a research-oriented evidence layer alongside model predictions.

---

### 07 — Explainable AI with SHAP

Apply **SHAP (SHapley Additive exPlanations)** to understand why a model produces a particular LSS prediction.

This would allow users to identify:

* Which planetary characteristics contribute most strongly
* Positive vs. negative feature contributions
* Feature-level explanations for individual exoplanets
* Global model behavior

The goal is to make the predictions **transparent and scientifically interpretable** rather than treating the model as a black box.

---

### 08 — Interactive LSS Dashboard

Develop an interactive dashboard for exploring the final system.

The dashboard could provide:

* LSS visualization
* Exoplanet ranking
* Individual exoplanet profiles
* Model predictions
* Feature importance
* SHAP explanations
* Scientific evidence retrieved through RAG
* Comparative analysis between exoplanets

This would provide a user-friendly interface for exploring the research framework.

---

# 🗺️ Planned End-to-End Architecture

The long-term system can evolve from the current preprocessing pipeline into:

```text
                    EXOPLANET DATA
                           │
                           ▼
                  Data Preprocessing
                           │
                           ▼
                Scientific Proxy Features
                           │
                           ▼
                    LSS Construction
                           │
                           ▼
                ┌─────────────────────┐
                │   ML Model Layer    │
                │                     │
                │ Regression Models   │
                │ Model Comparison    │
                │ Hyperparameter Opt. │
                │ K-Fold CV           │
                └──────────┬──────────┘
                           │
                           ▼
                 Predicted LSS Score
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
            SHAP          RAG       Exoplanet
         Explainability  Evidence     Ranking
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                  Interactive Dashboard
```

---

# 🧪 Current Status

| Component                            | Status          |
| ------------------------------------ | --------------- |
| Raw Data Processing                  | ✅ Implemented   |
| Feature Selection                    | ✅ Implemented   |
| Missing Value Handling               | ✅ Implemented   |
| Outlier Filtering                    | ✅ Implemented   |
| Feature Transformation               | ✅ Implemented   |
| VIF-Based Feature Reduction          | ✅ Implemented   |
| LSS Construction                     | ✅ Implemented   |
| Train/Test Split                     | ✅ Implemented   |
| Robust Scaling                       | ✅ Implemented   |
| ML Prediction Framework              | 🔭 Future Scope |
| Orbital/Stellar Interaction Modeling | 🔭 Future Scope |
| Hyperparameter Optimization          | 🔭 Future Scope |
| K-Fold Cross-Validation              | 🔭 Future Scope |
| RAG Scientific Evidence              | 🔭 Future Scope |
| SHAP Explainability                  | 🔭 Future Scope |
| Interactive Dashboard                | 🔭 Future Scope |

---

# 🛠️ Requirements

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

# 🚀 Usage

### 1. Clone the repository

```bash
git clone <repository-url>
cd Beyond-Binary-Predicting-Life-Sustainability-Scores-for-Exoplanets-
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn jupyter
```

### 3. Open the preprocessing notebook

```bash
jupyter notebook exoplanet_preprocessing.ipynb
```

### 4. Run the preprocessing pipeline

Execute the notebook cells sequentially to:

1. Load the raw dataset
2. Select relevant features
3. Handle missing values
4. Filter outliers
5. Transform features
6. Reduce multicollinearity
7. Construct the LSS
8. Generate train/test datasets
9. Apply robust scaling

---

# 📌 Research Direction

The central objective of this project is to move beyond a simplistic **binary definition of habitability** toward a more nuanced, continuous assessment framework.

The long-term vision is to combine:

**Scientific Indicators + Machine Learning + Explainable AI + Scientific Evidence Retrieval**

to create an interpretable framework for comparing the life-sustainability potential of known exoplanets.

---

## 🌌 Vision

> **From "Could this planet support life?" to "How strongly do its observed characteristics support life sustainability?"**

The current implementation establishes the **data and LSS foundation** required to pursue that question through machine learning and scientific analysis.

---

<div align="center">

**Beyond Binary — Exploring Life Sustainability Across Worlds.** 🌍 → 🪐

</div>
