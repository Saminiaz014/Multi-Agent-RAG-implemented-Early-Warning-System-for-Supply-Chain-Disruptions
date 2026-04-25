"""Entrypoint for the Supply Chain Decision Support System."""

import logging
import sys
from pathlib import Path

import yaml


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        path: Relative path to the settings file.

    Returns:
        Parsed configuration as a nested dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configure root logger for console and file output.

    Args:
        config: Full application config dictionary; reads config["logging"].
    """
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "logs/pipeline.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def main() -> None:
    """Run the DSS pipeline."""
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Pipeline initialized")
    print("Pipeline initialized")


if __name__ == "__main__":
    main()
