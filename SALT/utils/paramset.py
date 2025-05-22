import yaml
import torch

class Paramset():
    def __init__(self, config_yaml):
        self.config_yaml = config_yaml
        self.read_yaml(config_yaml)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def read_yaml(self, yaml_file):
        try:
            with open(yaml_file) as f:
                self.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e}: Config file not found") from e

