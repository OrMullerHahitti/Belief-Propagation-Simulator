import logging
import os

log_level = os.environ.get("BP_LOG_LEVEL", "CRITICAL").upper()
logging.getLogger().setLevel(getattr(logging, log_level, logging.CRITICAL))

for module_name in ["bp", "utils", "policies"]:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
