from types import SimpleNamespace

import yaml


def load_config(filepath='config.yaml'):
    with open(filepath) as f:
        config_dict = yaml.safe_load(f)

    # Convert nested dicts to SimpleNamespace for attribute access
    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [dict_to_ns(i) for i in d]
        return d

    return dict_to_ns(config_dict)
