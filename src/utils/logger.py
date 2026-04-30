"""
Professional logging utilities for Bird Call Detection project
Comprehensive logging with multiple handlers and configuration
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Color the level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logging(
    name: str = "bird_detection",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_logging: bool = True,
    file_logging: bool = True,
    json_logging: bool = False,
    max_bytes: int = 10*1024*1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console_logging: Enable console logging
        file_logging: Enable file logging
        json_logging: Enable JSON structured logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    from logging.handlers import RotatingFileHandler
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    
    json_formatter = JSONFormatter()
    
    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if file_logging and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Regular log file
        log_file = log_path / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_file = log_path / f"{name}_error.log"
        error_handler = RotatingFileHandler(
            error_file, maxBytes=max_bytes, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        # JSON log file (if enabled)
        if json_logging:
            json_file = log_path / f"{name}.json"
            json_handler = RotatingFileHandler(
                json_file, maxBytes=max_bytes, backupCount=backup_count
            )
            json_handler.setFormatter(json_formatter)
            logger.addHandler(json_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get or create a logger with standard configuration
    
    Args:
        name: Logger name (uses calling module if None)
        
    Returns:
        Configured logger
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(self, logger: logging.Logger, level: str = None, **extra_fields):
        self.logger = logger
        self.original_level = logger.level
        self.extra_fields = extra_fields
        
        if level:
            self.new_level = getattr(logging, level.upper())
        else:
            self.new_level = None
    
    def __enter__(self):
        if self.new_level:
            self.logger.setLevel(self.new_level)
        
        # Add extra fields to all log records in this context
        if self.extra_fields:
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                for key, value in self.extra_fields.items():
                    setattr(record, key, value)
                return record
            
            logging.setLogRecordFactory(record_factory)
            self.old_factory = old_factory
        else:
            self.old_factory = None
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer"""
        import time
        self.timers[name] = time.time()
        self.logger.debug(f"Timer '{name}' started")
    
    def stop_timer(self, name: str, log_level: str = "INFO") -> float:
        """Stop a named timer and log the duration"""
        import time
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' not found")
            return 0.0
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        level = getattr(logging, log_level.upper())
        self.logger.log(level, f"Timer '{name}' completed in {duration:.4f}s")
        
        return duration
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory usage {context}: "
                f"RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.1f}MB"
            )
        except ImportError:
            self.logger.warning("psutil not available for memory logging")
    
    def log_gpu_usage(self, context: str = ""):
        """Log GPU memory usage"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                cached = torch.cuda.memory_reserved() / 1024 / 1024
                
                self.logger.info(
                    f"GPU memory usage {context}: "
                    f"Allocated={allocated:.1f}MB, "
                    f"Cached={cached:.1f}MB"
                )
        except ImportError:
            pass


def configure_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging from configuration dictionary
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured logger
    """
    logging_config = config.get('logging', {})
    
    logger = setup_logging(
        name=config.get('project', {}).get('name', 'bird_detection'),
        level=logging_config.get('level', 'INFO'),
        log_dir=logging_config.get('log_dir', './logs'),
        console_logging=logging_config.get('console_logging', True),
        file_logging=logging_config.get('file_logging', True),
        json_logging=logging_config.get('json_logging', False)
    )
    
    return logger


# Context manager for timing operations
class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, name: str = "Operation", logger: logging.Logger = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.4f}s")


# Function decorator for logging function calls
def log_function_call(logger: logging.Logger = None, level: str = "DEBUG"):
    """Decorator to log function calls"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            log_level = getattr(logging, level.upper())
            
            func_logger.log(log_level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.log(log_level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                func_logger.error(f"{func.__name__} failed with error: {e}")
                raise
        
        return wrapper
    return decorator


def setup_project_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging for the entire project
    
    Args:
        config: Project configuration dictionary
        
    Returns:
        Main project logger
    """
    # Configure main logger
    main_logger = configure_logging_from_config(config)
    
    # Configure module-specific loggers
    module_loggers = {
        'src.data.xeno_canto_api': 'INFO',
        'src.data.audio_processor': 'INFO',
        'src.models.efficientnet_classifier': 'INFO',
        'src.training.trainer': 'INFO',
        'src.inference.realtime_detector': 'INFO'
    }
    
    for module_name, level in module_loggers.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, level.upper()))
    
    main_logger.info("Project logging configured successfully")
    return main_logger