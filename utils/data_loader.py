import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(config, split=False, test_size=0.2, val_size=0.1):
    """
    Load data from a file, separate features and target, and optionally split into train, validation, and test sets.

    :param config: Configuration dictionary with file path.
    :param target_column: Name of the target variable column.
    :param split: Boolean, whether to split the data or not.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the training dataset to include in the validation split.
    :return: Depending on the 'split', returns either full dataset or split datasets along with their targets.
    """
    # Get the file path from the config file
    file_path = config["dataset"]["data_path"]

    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop("RiskPerformance", axis=1)
    y = data["RiskPerformance"]

    if split:
        # Calculate validation size with respect to the original dataset
        val_size_adjusted = val_size / (1 - test_size)
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    return X, y

