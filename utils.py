import yaml

def get_yaml(path):
    """
    Load yaml file 
    """
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    return data