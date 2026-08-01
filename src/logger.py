import logging
import os

logger = logging.getLogger("bottuyensinh")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

logger.setLevel(os.getenv("LOG_LEVEL", "INFO").strip().upper())
