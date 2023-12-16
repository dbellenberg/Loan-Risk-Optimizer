import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve


def load_config():
    """
    Load the project configuration from a YAML file.

    :param config_file: The path to the YAML configuration file.
    :return: The configuration dictionary.
    """
    # Ensuring that working directory is appropriately set
    assert os.getcwd().split('/')[-1] == 'Loan-Risk-Optimizer', 'Working directory not set to root of project. Currently working in: ' + os.getcwd()

    # Loading the config file
    with open("config.yml") as file:
        config = yaml.safe_load(file)
    
    return config

# Evaluation function that prints classification report and confusion matrix
def evaluate(X_test, y_test, model):

    #predict on validation set
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\033[34m{model.__class__.__name__}\033[0m")

    print(f" \033[32mClassification Report:\033[0m")
    print(classification_report(y_test, y_pred))

    print(f" \033[32mAccuracy:\033[0m {accuracy_score(y_test, y_pred):.2f}")
    print()

    print(f" \033[32mConfusion Matrix:\033[0m")
    print(cm, flush=True)
    print()
 

