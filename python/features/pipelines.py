import numpy as np
import pandas as pd
try:
    import talib  # type: ignore
    import talib.abstract as ta
except Exception:  # pragma: no cover - talib optional
    talib = None
    ta = None


def _talib_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all TA-Lib indicators available for the dataframe."""
    if ta is None:
        return pd.DataFrame(index=df.index)

    inputs = {
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "close": df["Close"],
        "volume": df.get("Volume"),
    }
    features = pd.DataFrame(index=df.index)
    for name in ta.get_functions():
        func = ta.Function(name)
        try:
            out = func(inputs)
        except Exception:
            continue
        if isinstance(out, pd.DataFrame):
            out = out.add_prefix(f"{name}_")
            features = features.join(out)
        else:
            features[name] = out
    return features


def _price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic price action statistics."""
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    close = df["Close"]
    range_ = (high - low).replace(0, np.nan)
    upper_wick = high - np.fmax(open_, close)
    lower_wick = np.fmin(open_, close) - low
    feats = pd.DataFrame(index=df.index)
    feats["wick_upper"] = upper_wick
    feats["wick_lower"] = lower_wick
    feats["wick_ratio_upper"] = upper_wick / range_
    feats["wick_ratio_lower"] = lower_wick / range_
    atr = talib.ATR(high, low, close) if talib is not None else pd.Series([0] * len(df), index=df.index)
    feats["ATR"] = atr
    return feats


def _window_embeddings(returns: pd.Series, length: int) -> pd.Series:
    """Create rolling windows of normalized returns."""
    arrs = []
    for i in range(len(returns)):
        if i + 1 < length:
            arrs.append(np.array([np.nan] * length))
            continue
        window = returns.iloc[i - length + 1 : i + 1].values
        std = window.std()
        if std == 0 or np.isnan(std):
            std = 1.0
        normed = (window - window.mean()) / std
        arrs.append(normed)
    return pd.Series(arrs, index=returns.index)


def _datetime_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract calendar information from the index."""
    feats = pd.DataFrame(index=index)
    feats["weekday"] = index.dayofweek
    feats["month_end"] = index.is_month_end.astype(int)
    try:
        import pandas_market_calendars as mcal  # type: ignore

        cal = mcal.get_calendar("EUREX")
        start = index.min().date()
        end = index.max().date()
        valid = cal.valid_days(start_date=start, end_date=end)
        valid_dates = set(valid.tz_convert(None).normalize().date)
        idx_dates = index.tz_localize(None).normalize().date
        feats["holiday_flag"] = (~pd.Series(idx_dates, index=index).isin(valid_dates)).astype(int)
    except Exception:  # pragma: no cover - calendar optional
        feats["holiday_flag"] = 0
    return feats


def compute_features(
    df: pd.DataFrame, exogenous: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute a wide feature set for a price dataframe."""
    features = pd.DataFrame(index=df.index)
    features = features.join(_talib_features(df))
    features = features.join(_price_action_features(df))
    returns = df["Close"].pct_change().fillna(0)
    features["emb_32"] = _window_embeddings(returns, 32)
    features["emb_64"] = _window_embeddings(returns, 64)
    features = features.join(_datetime_features(df.index))
    if exogenous is not None and "Close" in exogenous:
        aligned = exogenous["Close"].reindex(df.index, method="ffill")
        features["dax_future_spread"] = df["Close"] - aligned
    return features
