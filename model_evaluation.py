"""Evaluates the trained model on the original data set,
and the dataset after logarithmic/inverse/sqrt transforms.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_transform_data(file_path):
    """Load housing data and prepare features with various target transformations.

    Args:
        file_path: str. Path to the CSV file containing the data.

    Returns:
        A tuple of:
        - features_df: pd.DataFrame. The feature columns.
        - transformed_targets: dict. Dictionary containing various target transformations.
    """
    data_df = pd.read_csv(file_path)
    features_df = data_df[['median_salary', 'population_size', 'number_of_jobs', 'area_size']]
    transformed_targets = {
        'Original': data_df['mean_salary'],
        'Log Transformation': np.log1p(data_df['mean_salary']),
        'Square Root Transformation': np.sqrt(data_df['mean_salary']),
        'Inverse Transformation': 1 / (data_df['mean_salary'] + 1e-6)
    }
    return features_df, transformed_targets

def scale_data(x_train, x_test):
    """Scale features using StandardScaler.

    Args:
        x_train: np.ndarray. The training feature data.
        x_test: np.ndarray. The test feature data.

    Returns:
        Tuple of:
        - scaled_train_data: np.ndarray. Scaled training feature data.
        - scaled_test_data: np.ndarray. Scaled test feature data.
    """
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(x_train)
    scaled_test_data = scaler.transform(x_test)
    return scaled_train_data, scaled_test_data

def perform_hyperparameter_search(x_train, y_train):
    """Perform RandomizedSearchCV to find the best hyperparameters.

    Args:
        x_train: np.ndarray. The training feature data.
        y_train: pd.Series. The training target values.

    Returns:
        RandomizedSearchCV. The search object containing the best model.
    """
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(x_train, y_train)
    return search

def train_and_evaluate_model(features_df, target_series, transformation_name, results_dict):
    """Train and evaluate a RandomForestRegressor model.

    Args:
        features_df: pd.DataFrame. The feature columns.
        target_series: pd.Series. The target values.
        transformation_name: str. Name of the target transformation.
        results_dict: dict. Dictionary to store evaluation metrics and best parameters.

    Returns:
        None. Updates the results_dict with evaluation metrics.
    """
    # Drop missing target values and align features
    cleaned_target = target_series.dropna()
    aligned_features = features_df.loc[cleaned_target.index]

    # Split training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        aligned_features, cleaned_target, test_size=0.2, random_state=42
    )

    # Scale features
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # Perform hyperparameter search
    search = perform_hyperparameter_search(x_train_scaled, y_train)
    best_model = search.best_estimator_

    # Make predictions
    predictions = best_model.predict(x_test_scaled)

    # Compute and store metrics
    metrics = {
        'MSE': mean_squared_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'R²': r2_score(y_test, predictions),
        'Best Parameters': search.best_params_
    }

    results_dict[transformation_name] = metrics

def main():
    """Main function to load data, train models, and display results."""
    features_df, transformed_targets = load_and_transform_data('data/final_housing_data.csv')

    results = {}

    for name, target in transformed_targets.items():
        train_and_evaluate_model(features_df, target, name, results)

    for transformation, metrics in results.items():
        print(f"Results for {transformation}:")
        print(f"  Best Parameters: {metrics['Best Parameters']}")
        print(f"  Mean Squared Error (MSE): {metrics['MSE']:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"  Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
        print(f"  R-squared (R²): {metrics['R²']:.4f}\n")

if __name__ == '__main__':
    main()
