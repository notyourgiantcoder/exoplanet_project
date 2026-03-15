# Exoplanet Project

This project focuses on the analysis and preprocessing of exoplanet data to prepare it for machine learning applications. The main goal is to compute a Life Support Score (LSS) for various exoplanets based on their characteristics.

## Project Structure

The project consists of the following files:

- **exoplanet_preprocessing.ipynb**: A Jupyter Notebook that contains the data preprocessing steps for exoplanet data. It includes:
  - Loading raw data
  - Selecting core columns
  - Imputing missing values
  - Applying outlier filters
  - Performing log transformations
  - Reducing multicollinearity
  - Computing the LSS target
  - Splitting the dataset for machine learning

- **PSCompPars_2026.03.05_02.02.54.csv**: The raw data on exoplanets used as input for preprocessing.

- **exoplanets_raw_backup.csv**: A backup of the raw exoplanet data before any preprocessing steps are applied.

- **exoplanets_selected.csv**: Contains the selected core columns from the raw data after the initial selection step.

- **exoplanets_step3_clean.csv**: The cleaned dataset after imputing missing values.

- **exoplanets_step4_clean.csv**: The dataset after applying outlier filters.

- **exoplanets_step5_transformed.csv**: The dataset after applying log transformations to certain features.

- **exoplanets_step6_final_features.csv**: The final set of features after performing variance inflation factor (VIF) reduction.

- **exoplanets_step7_model_ready.csv**: The dataset prepared for machine learning, including the computed LSS target.

- **exoplanets_step7_with_lss.csv**: The dataset with the LSS scores added for each exoplanet.

- **transform_metadata.json**: Contains metadata about the transformations applied to the dataset, including which features were log-transformed.

- **robust_scaler.pkl**: The trained RobustScaler object used for scaling the features in the dataset.

- **X_train.csv**: Contains the training features for the machine learning model.

- **X_test.csv**: Contains the testing features for the machine learning model.

- **X_train_scaled.csv**: Contains the scaled training features.

- **X_test_scaled.csv**: Contains the scaled testing features.

- **y_train.csv**: Contains the training target values (LSS).

- **y_test.csv**: Contains the testing target values (LSS).

## Usage

1. **Data Preprocessing**: Open the `exoplanet_preprocessing.ipynb` notebook to run the preprocessing steps.
2. **Model Training**: Use the generated training and testing datasets (`X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`) for training machine learning models.
3. **Scaling**: The features are scaled using the `robust_scaler.pkl` to ensure better performance of the models.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## License

This project is licensed under the MIT License. See the LICENSE file for details.
