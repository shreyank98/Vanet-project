"""
Logging utilities for VANET IDS project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from utils.config import Config

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_default_logger():
    """Get default logger for the project"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Config.LOGS_DIR / f'vanet_ids_{timestamp}.log'
    return setup_logger('VANET_IDS', log_file)
