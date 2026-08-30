import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "HeliosEngine", log_dir: str = "logs") -> logging.Logger:
    """
    Configures an enterprise-grade rotating file logger.
    Outputs to both the console and a secure log file to prevent data loss on UI crash.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    
    # Prevent adding duplicate handlers if instantiated multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | [%(module)s:%(funcName)s] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. File Handler (Rotates after 5MB, keeps last 3 backups)
        log_file = os.path.join(log_dir, "helios_execution.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # 2. Console Handler (for CLI execution)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING) # Only print warnings/errors to console
        logger.addHandler(console_handler)

    return logger