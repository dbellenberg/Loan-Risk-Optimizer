import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import seaborn as sns

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
def evaluate(y_test, y_pred, model_name):
    #print accuracy
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print()
    cm = confusion_matrix(y_test, y_pred)

    print(f"\033[34m{model_name}\033[0m")
    print()
    print(f" \033[32mClassification Report:\033[0m")
    print(classification_report(y_test, y_pred))
    print(f" \033[32mConfusion Matrix:\033[0m")
    
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(f'{model_name}: Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()