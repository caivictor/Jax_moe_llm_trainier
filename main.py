import argparse
import yaml
from types import SimpleNamespace # To easily access config values like config.data.batch_size
from training.trainer import train # Adjusted import path

def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # Convert dict to SimpleNamespace for easier attribute access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        return d
    return dict_to_namespace(config_dict)

def main():
    parser = argparse.ArgumentParser(description="Train a JAX MoE Language Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml", # Default config path
        help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:")
    # Simple print of top-level keys
    print(yaml.dump(config.__dict__, default_flow_style=False))


    # Start training
    train(config)

if __name__ == "__main__":
    main()