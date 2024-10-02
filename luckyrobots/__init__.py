from importlib.metadata import PackageNotFoundError, version

from .core import start, send_message
from .event_handler import on
from .comms import create_instructions, run_server

__all__ = ["start", "send_message", "on", "create_instructions", "run_server"]


try:
    __version__ = version("luckyrobots")
except PackageNotFoundError:
    __version__ = "unknown"