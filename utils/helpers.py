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


def format_classification_report(y_true, y_pred):
    # Generate the classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Format the report
    formatted_report = "class        precision    recall  f1-score   support\n\n"
    for label, metrics in report.items():
        if label in ['macro avg', 'weighted avg']:
            # Format macro and weighted averages
            line = f"{label:<12} {metrics['precision']:>9.3f} {metrics['recall']:>7.3f} {metrics['f1-score']:>9.3f} {metrics['support']:>8}\n"
        elif label == 'accuracy':
            # Handle the accuracy case separately
            line = f"\n{label:<12} {metrics:>9.3f}\n"
        else:
            # Format individual class metrics
            line = f"{label:<12} {metrics['precision']:>9.3f} {metrics['recall']:>7.3f} {metrics['f1-score']:>9.3f} {metrics['support']:>8}\n"
        formatted_report += line

    return formatted_report


# Evaluation function that prints classification report and confusion matrix
def evaluate(X_test, y_test, model):

    #predict on validation set
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\033[34m{model.__class__.__name__}\033[0m")

    print(f" \033[32mClassification Report:\033[0m")
    #print(classification_report(y_test, y_pred))
    print(format_classification_report(y_test, y_pred))


    print(f" \033[32mConfusion Matrix:\033[0m")
    print(cm, flush=True)
    print()
 

