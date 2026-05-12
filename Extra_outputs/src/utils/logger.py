"""
logger.py
Rich-powered logger so console output is readable and colourful.
"""

from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)
