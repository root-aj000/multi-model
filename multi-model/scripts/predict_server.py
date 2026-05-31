"""
Prediction server startup script.

Starts the FastAPI prediction server with configurable host, port,
and model configuration.
"""

import argparse
import logging

import uvicorn

from app.app import app
from lib.utils.config import load_config
from lib.utils.lifecycle import setup_directories

logger = logging.getLogger(__name__)

# Default server host
DEFAULT_HOST = "0.0.0.0"

# Default server port
DEFAULT_PORT = 8000

# Default configuration file path
DEFAULT_CONFIG_PATH = "configs/model/model_config.json"


def main() -> None:
    """
    Start the prediction server.

    Parses command-line arguments, sets up directories, and launches
    the uvicorn server with the FastAPI application.
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
        default=DEFAULT_HOST,
        help="Host to bind the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to run the server on",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    setup_directories()

    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
