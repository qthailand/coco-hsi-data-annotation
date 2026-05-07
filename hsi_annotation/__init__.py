import logging

from .app import run

__all__ = ["run"]

logging.getLogger(__name__).addHandler(logging.NullHandler())