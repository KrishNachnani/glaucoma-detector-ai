import os
import yaml
from loguru import logger

def setup_logging():
    """
    Configure logger based on config.yaml settings.
    Returns the configured logger instance.
    """
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Configure logger
    log_config = config["logging"]
    os.makedirs(os.path.dirname(log_config["file_path"]), exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Configure handlers based on config.yaml
    # File handler
    logger.configure(
        handlers=[
            {
                "sink": log_config["file_path"],
                "level": log_config["level"],
                "format": log_config["format"],
                "rotation": log_config["rotation"],
                "retention": log_config["retention"],
                "compression": log_config["compression"],
                "enqueue": log_config.get("enqueue", True),
                "backtrace": log_config.get("backtrace", True),
                "diagnose": log_config.get("diagnose", False),
                "serialize": log_config.get("serialize", False),
            }
        ]
    )
    
    # Add console handler if enabled
    if log_config.get("console", True):
        logger.add(
            sink=lambda msg: print(msg),
            level=log_config["level"],
            format=log_config["format"],
            enqueue=log_config.get("enqueue", True),
            backtrace=log_config.get("backtrace", True),
            diagnose=log_config.get("diagnose", False),
        )
    
    return logger

# Initialize logger when module is imported
logger = setup_logging()