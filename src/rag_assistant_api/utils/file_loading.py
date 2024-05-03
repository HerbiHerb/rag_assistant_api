import yaml


def load_yaml_file(yaml_file_fp: str):
    with open(yaml_file_fp, "r") as file:
        config_data = yaml.safe_load(file)
        return config_data
