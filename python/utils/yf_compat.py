import inspect
import os

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

_COMPAT_ARGS: dict[str, bool] = {"progress": False}
if yf is not None and "threads" in inspect.signature(yf.download).parameters:
    _COMPAT_ARGS["threads"] = False

# ensure yfinance does not use the default SQLite cache
os.environ.setdefault("YFINANCE_NO_CACHE", "1")

try:
    from yfinance.exceptions import YFPricesMissingError  # type: ignore
except Exception:  # pragma: no cover - fallback for tests without yfinance
    class YFPricesMissingError(Exception):
        """Raised when yfinance returns no data."""
        pass
