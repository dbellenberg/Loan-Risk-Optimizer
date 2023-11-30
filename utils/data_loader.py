import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(config,split=False, test_size=0.2, val_size=0.1):
    """
    Load data from a file and optionally split into train, validation, and test sets.

    :param file_path: Path to the data file.
    :param split: Boolean, whether to split the data or not.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the training dataset to include in the validation split.
    :return: Depending on the 'split', returns either full dataset or split datasets.
    """
    #path /Users/davidbellenberg/github_projects/Loan-Risk-Optimizer/data/heloc.csv

    # Get the file path from the config file

    file_path = config["dataset"]["data_path"]

    # Load the dataset
    data = pd.read_csv(file_path)

    if split:
        # Calculate validation size with respect to the original dataset
        val_size_adjusted = val_size / (1 - test_size)
        
        # Splitting the data
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size_adjusted, random_state=42)

        return train_data, val_data, test_data

    return data
