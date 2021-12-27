import yaml
import pathlib


def load_config(path: pathlib.Path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

