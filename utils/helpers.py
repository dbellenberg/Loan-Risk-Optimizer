import os
import yaml

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
