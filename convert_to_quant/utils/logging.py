import logging
import sys
import functools

# Custom Levels
MINIMAL_LEVEL = 30
NORMAL_LEVEL = 20
VERBOSE_LEVEL = 15
DEBUG_LEVEL = 10


class LazyString:
    """
    Lazy string evaluation wrapper for logging.
    
    Usage:
        debug("Processing tensor: %s", LazyString(lambda: expensive_operation()))
        
    The lambda is only called if the log level is enabled.
    """
    def __init__(self, fn):
        self.fn = fn
        
    def __str__(self):
        return str(self.fn())
    
    def __repr__(self):
        return repr(self.fn())


class LazyFormat:
    """
    Lazy format string evaluation for logging.
    
    Usage:
        debug(LazyFormat("Processing {name}", name=lambda: get_name()))
        
    Lambdas in kwargs are only called if the log level is enabled.
    """
    def __init__(self, fmt: str, **kwargs):
        self.fmt = fmt
        self.kwargs = kwargs
        
    def __str__(self):
        resolved = {k: (v() if callable(v) else v) for k, v in self.kwargs.items()}
        return self.fmt.format(**resolved)


def lazy_debug(fn):
    """
    Decorator for functions that return debug strings.
    Only evaluates if DEBUG level is enabled.
    """
    def wrapper(*args, **kwargs):
        if not get_logger().isEnabledFor(DEBUG_LEVEL):
            return ""
        return fn(*args, **kwargs)
    return wrapper

logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")
logging.addLevelName(MINIMAL_LEVEL, "MINIMAL")

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno <= DEBUG_LEVEL:
            self._style._fmt = "[%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
        elif record.levelno <= VERBOSE_LEVEL:
            self._style._fmt = "%(message)s"
        elif record.levelno <= NORMAL_LEVEL:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "[%(levelname)s] %(message)s"
        return super().format(record)

def setup_logging(verbose_arg: str = "NORMAL"):
    level_map = {"DEBUG": DEBUG_LEVEL, "VERBOSE": VERBOSE_LEVEL, "NORMAL": NORMAL_LEVEL, "MINIMAL": MINIMAL_LEVEL}
    level = level_map.get(verbose_arg.upper(), NORMAL_LEVEL)
    logger = logging.getLogger("convert_to_quant")
    logger.setLevel(level)
    if logger.handlers: logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    return logger

def get_logger(): return logging.getLogger("convert_to_quant")

def log_debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        if logger.isEnabledFor(DEBUG_LEVEL):
            logger.log(DEBUG_LEVEL, f"CALL {func.__name__}")
        result = func(*args, **kwargs)
        if logger.isEnabledFor(DEBUG_LEVEL):
            logger.log(DEBUG_LEVEL, f"RET {func.__name__}")
        return result
    return wrapper

def debug(msg, *args, **kwargs): get_logger().log(DEBUG_LEVEL, msg, *args, **kwargs)
def verbose(msg, *args, **kwargs): get_logger().log(VERBOSE_LEVEL, msg, *args, **kwargs)
def normal(msg, *args, **kwargs): get_logger().log(NORMAL_LEVEL, msg, *args, **kwargs)
def info(msg, *args, **kwargs): normal(msg, *args, **kwargs)
def minimal(msg, *args, **kwargs): get_logger().log(MINIMAL_LEVEL, msg, *args, **kwargs)
def warning(msg, *args, **kwargs): get_logger().warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs): get_logger().error(msg, *args, **kwargs)
