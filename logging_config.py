"""
logging_config.py - Structured JSON Logging with Trace ID Support
ControlPlane.ai (PS1 Architecture)

Provides centralized logging configuration for the pipeline:
1. JSON-formatted structured log output for machine parsing.
2. Trace ID / Request ID propagation via contextvars for cross-stage correlation.
3. Configurable log levels via CONTROLPLANE_LOG_LEVEL environment variable.
"""

import os
import json
import logging
import datetime
import uuid
import contextvars
from typing import Optional


# Context variable for per-request trace ID propagation
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Returns the current request's trace ID, or empty string if not set."""
    return _trace_id.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    Sets (or generates) a trace ID for the current request context.
    Returns the trace ID that was set.
    """
    tid = trace_id or f"req-{uuid.uuid4().hex[:12]}"
    _trace_id.set(tid)
    return tid


def clear_trace_id() -> None:
    """Clears the trace ID for the current context."""
    _trace_id.set("")


class StructuredJSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects for structured log aggregation.
    Includes timestamp, level, logger name, message, trace_id, and any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra structured fields passed via `extra=` kwarg
        for key in ("stage", "component", "latency_ms", "decision", "score"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class TraceIdFilter(logging.Filter):
    """Injects ContextVar trace_id into standard LogRecord attributes."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "none"
        return True


def configure_logging(
    level: Optional[str] = None,
    json_output: bool = True
) -> None:
    """
    Configures the root logger for the ControlPlane pipeline.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). 
               Defaults to CONTROLPLANE_LOG_LEVEL env var, or INFO.
        json_output: If True, uses structured JSON formatter. 
                     If False, uses human-readable format.
    """
    log_level = level or os.environ.get("CONTROLPLANE_LOG_LEVEL", "INFO")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicate output
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(TraceIdFilter())

    if json_output:
        handler.setFormatter(StructuredJSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(trace_id)s): %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        ))

    root_logger.addHandler(handler)
