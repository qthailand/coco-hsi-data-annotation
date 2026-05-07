import sys
import logging

from hsi_annotation.app import run

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
logging.getLogger("spectral").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

if __name__ == "__main__":
    sys.exit(run())
