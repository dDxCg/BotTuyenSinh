import logging
import os

_RESET = "\033[0m"
_TIME_COLOR = "\033[32m"  # xanh lá

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[34m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[31m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        time_str = f"{_TIME_COLOR}{self.formatTime(record)}{_RESET}"
        level_color = _LEVEL_COLORS.get(record.levelno, "")
        level_str = f"{level_color}{record.levelname}{_RESET}" if level_color else record.levelname
        return f"{time_str} {level_str} {record.name}: {record.getMessage()}"


logger = logging.getLogger("bottuyensinh")

if not logger.handlers:
    if os.name == "nt":
        os.system("")  # bật ANSI escape trên cmd/PowerShell cũ
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
    logger.propagate = False

logger.setLevel(os.getenv("LOG_LEVEL", "INFO").strip().upper())
