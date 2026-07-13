import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters successfully loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise