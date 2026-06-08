"""
Prediction server startup script.

Starts the FastAPI prediction server with configurable host, port,
and model configuration. Host and port default to values in model_config.json
(server.host / server.port) and can be overridden via CLI flags.
"""

import argparse
import logging

import uvicorn

from app.app import app
from lib.utils.config import load_config
from lib.utils.lifecycle import setup_directories

logger = logging.getLogger(__name__)

# Path to the default configuration file (CLI default only)
DEFAULT_CONFIG_PATH = "configs/model/model_config.json"


def main() -> None:
    """
    Start the prediction server.

    Parses command-line arguments, sets up directories, and launches
    the uvicorn server with the FastAPI application.
    Host and port are read from config (server.host / server.port)
    unless overridden by CLI flags.
    """
    parser = argparse.ArgumentParser(
        description="Start the prediction server",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server (overrides config server.host)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (overrides config server.port)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    setup_directories()

    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 8000)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
