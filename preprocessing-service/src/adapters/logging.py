"""
Adapters for logging operations.
"""
import logging
from src.domain.ports import ILogger


class PythonLogger(ILogger):
    """
    Logger adapter using Python's built-in logging module.
    """
    
    def __init__(self, name: str = "preprocessing-service"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str, exception: Exception = None):
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


class ConsoleLogger(ILogger):
    """
    Simple console logger for development/testing.
    """
    
    def info(self, message: str):
        print(f"[INFO] {message}")
    
    def warning(self, message: str):
        print(f"[WARNING] {message}")
    
    def error(self, message: str, exception: Exception = None):
        print(f"[ERROR] {message}")
        if exception:
            print(f"  Exception: {str(exception)}")
    
    def debug(self, message: str):
        print(f"[DEBUG] {message}")