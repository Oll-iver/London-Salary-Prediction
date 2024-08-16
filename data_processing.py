"""Processes the CSV files in /data/ and creates
final_housing_data.csv, which we will train the model on."""
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

def load_and_merge_data():
    """Loads and merges the necessary datasets, and processes them.

    Returns:
        merged_df: pd.DataFrame. The merged and cleaned dataset.
    """
    # Load datasets
    yearly_data = pd.read_csv('data/housing_in_london_yearly_variables.csv')
    borough_profiles = pd.read_csv('data/london-borough-profiles-2016 Data set.csv')

    # Normalise area names for merging
    yearly_data['area'] = yearly_data['area'].str.strip().str.lower()
    borough_profiles['Area name'] = borough_profiles['Area name'].str.strip().str.lower()

    # Merge datasets on the area name
    merged_df = pd.merge(yearly_data, borough_profiles, how='inner',
                         left_on='area', right_on='Area name')

    # Convert nonnumerical columns to numeric types
    merged_df['mean_salary'] = pd.to_numeric(merged_df['mean_salary'], errors='coerce')
    merged_df['recycling_pct'] = pd.to_numeric(merged_df['recycling_pct'], errors='coerce')

    return merged_df

def feature_selection(data_df):
    """Selects important features using Recursive Feature Elimination.

    Args:
        data_df: pd.DataFrame. The merged and cleaned dataset.

    Returns:
        selected_features_df: pd.DataFrame. The dataset with selected features.
        target_series: pd.Series. The target variable.
    """
    # Define features and target
    feature_columns = ['median_salary', 'population_size', 'number_of_jobs', 'area_size']
    features_df = data_df[feature_columns]
    target_series = data_df['mean_salary'].dropna()
    features_df = features_df.loc[target_series.index]  # Align features with the cleaned target

    # Perform feature selection with RFE
    rf_model = RandomForestRegressor(random_state=42)
    rfe_selector = RFE(estimator=rf_model, n_features_to_select=4)
    rfe_selector.fit(features_df, target_series)
    selected_feature_names = features_df.columns[rfe_selector.support_]

    selected_features_df = features_df[selected_feature_names]

    return selected_features_df, target_series

def save_processed_data(features_df, target_series, output_file='data/final_housing_data.csv'):
    """Saves the selected features and target to a CSV file.

    Args:
        features_df: pd.DataFrame. The dataset with selected features.
        target_series: pd.Series. The target variable.
        output_file: str. The file path to save the final dataset.

    Returns:
        None
    """
    final_df = pd.concat([features_df, target_series], axis=1)
    final_df.to_csv(output_file, index=False)

def main():
    """Main function to load data, select features, and save the final dataset."""
    merged_df = load_and_merge_data()
    selected_features_df, target_series = feature_selection(merged_df)
    save_processed_data(selected_features_df, target_series)

if __name__ == '__main__':
    main()
