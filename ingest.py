import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 3600  # 1 hour

def fetch_yfinance_with_retry(ticker: str, interval: str, period: str = "7d", start=None, end=None, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            if start and end:
                df = t.history(interval=interval, start=start, end=end)
            else:
                df = t.history(interval=interval, period=period)
            if not df.empty:
                # Ensure timezone aware index (UTC)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
                return df
        except Exception as e:
            time.sleep(1 + attempt * 2)
    return pd.DataFrame()

def get_market_data(ticker: str, interval: str = "1m", start=None, end=None) -> pd.DataFrame:
    # Use SPX index standard representation for yfinance if ^SPX is passed
    yf_ticker = ticker
    
    safe_ticker = yf_ticker.replace("^", "")
    suffix = f"_{start}_{end}" if start and end else ""
    cache_file = os.path.join(CACHE_DIR, f"{safe_ticker}_{interval}{suffix}.parquet")
    
    # Check cache TTL
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < CACHE_TTL_SECONDS:
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    # In pyarrow/fastparquet sometimes index loses tz when written, ensure UTC
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    return df
            except Exception:
                pass # Fallback to fetch
                
    # Fetch fresh
    # Period for 1m bars in yfinance is max 7d
    period = "7d" if interval == "1m" else "60d"
    df = fetch_yfinance_with_retry(yf_ticker, interval, period, start, end)
    
    if not df.empty:
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass # Cache write fail shouldn't break app
            
    return df

def get_vix_for_day(date_obj: datetime) -> float:
    # 1 day cache effectively
    cache_file = os.path.join(CACHE_DIR, "VIX_daily.parquet")
    df = pd.DataFrame()
    target_dt = pd.to_datetime(date_obj, utc=True)
    
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < CACHE_TTL_SECONDS:
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    min_idx = df.index.min()
                    if min_idx > target_dt:
                        df = pd.DataFrame()
            except (OSError, ValueError):
                pass
                
    if df.empty:
        df = fetch_yfinance_with_retry("^VIX", "1d", "max")
        if not df.empty:
            df.to_parquet(cache_file)
            
    if df.empty:
        return 0.0
        
    # Find closest previous close
    valid_dates = df[df.index <= target_dt]
    if not valid_dates.empty:
        return float(valid_dates.iloc[-1]['Close'])
    return 0.0

def _get_annualization_factor(interval: str) -> float:
    # Approx 252 trading days, 390 mins proper day
    if interval == "1m":
        return 252 * 390
    elif interval == "5m":
        return 252 * (390 / 5)
    elif interval == "15m":
        return 252 * (390 / 15)
    elif interval == "60m" or interval == "1h":
        return 252 * (390 / 60)
    return 252

def compute_realized_vol(df: pd.DataFrame, end_time: datetime, window_mins: int, interval: str="1m") -> float:
    if df.empty:
        return 0.0
        
    start_time = end_time - timedelta(minutes=window_mins)
    
    # Filter
    mask = (df.index >= pd.to_datetime(start_time, utc=True)) & (df.index <= pd.to_datetime(end_time, utc=True))
    window_df = df.loc[mask].copy()
    
    if len(window_df) < 2:
        return 0.0
        
    window_df['log_ret'] = np.log(window_df['Close'] / window_df['Close'].shift(1))
    std_dev = window_df['log_ret'].std()
    
    if pd.isna(std_dev):
        return 0.0
        
    ann_factor = np.sqrt(_get_annualization_factor(interval))
    return float(std_dev * ann_factor)

def import_trades_csv(file_path_or_buffer) -> pd.DataFrame:
    """
    Generic CSV parser for broker exports (e.g. Robinhood).
    More forgiving on bad lines so uploads don't crash the app.
    """
    try:
        # engine='python' + on_bad_lines='skip' is robust to stray commas / malformed rows
        df = pd.read_csv(file_path_or_buffer, engine="python", on_bad_lines="skip")
    except Exception as e:
        # Surface a clear error to the UI instead of a low-level parser error
        raise ValueError(f"Failed to parse CSV. Please check the file format. Underlying error: {e}") from e
    # Expected user mapping done in UI. Here we return raw for UI to present.
    return df


def filter_option_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only option trades (rows where Description or Instrument contains 'call' or 'put').
    Returns filtered DataFrame, or original if no option rows found.
    """
    if df.empty:
        return df
    col_map = {c.lower(): c for c in df.columns}
    desc_col = col_map.get("description") or col_map.get("instrument")
    if not desc_col:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        desc_col = text_cols[0] if text_cols else df.columns[0]
    desc_series = df[desc_col].astype(str)
    mask = desc_series.str.contains("call", case=False, na=False) | desc_series.str.contains("put", case=False, na=False)
    opt_df = df[mask].copy()
    return opt_df if not opt_df.empty else df


import re
from typing import List, Dict, Any, Optional


def _parse_option_description(desc: str, instrument: str) -> Optional[Dict[str, Any]]:
    """
    Parse Robinhood-style option description into ticker, expiry, option_type, strike.
    Handles: "CRCL 6/27/2025 Put $202.50", "SPY 05/15/2024 515.00 Call", "Option Expiration for SPY 2/4/"
    """
    if not desc or pd.isna(desc):
        return None
    desc = str(desc).strip()
    # Try: INSTRUMENT M/D/YYYY Put $STRIKE or INSTRUMENT M/D/YYYY Call $STRIKE
    m = re.search(r"(\w+)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(Put|Call)\s+\$?([\d.]+)", desc, re.I)
    if m:
        return {"ticker": m.group(1), "expiry_str": m.group(2), "option_type": m.group(3).lower(), "strike": float(m.group(4))}
    # Try: INSTRUMENT MM/DD/YYYY STRIKE Call/Put (strike before type)
    m = re.search(r"(\w+)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+([\d.]+)\s+(Call|Put)", desc, re.I)
    if m:
        return {"ticker": m.group(1), "expiry_str": m.group(2), "option_type": m.group(4).lower(), "strike": float(m.group(3))}
    # Fallback: use Instrument as ticker if we can at least get type and strike
    m = re.search(r"(Put|Call)\s+\$?([\d.]+)", desc, re.I)
    if m:
        return {"ticker": str(instrument).strip() if instrument else "UNK", "expiry_str": "", "option_type": m.group(1).lower(), "strike": float(m.group(2))}
    return None


def _parse_quantity(qty) -> int:
    """Extract contract count from Quantity (e.g. '1', '1.0', '1S', '2')."""
    if qty is None or pd.isna(qty):
        return 1
    try:
        return max(1, int(float(qty)))
    except (ValueError, TypeError):
        s = str(qty).strip()
        m = re.search(r'([\d.]+)', s)
        if m:
            try:
                return max(1, int(float(m.group(1))))
            except ValueError:
                pass
        return 1


def _parse_price(price) -> float:
    """Parse price/amount; OEXP/None -> 0. Handles $, commas, and parenthesis for negatives."""
    if price is None or pd.isna(price):
        return 0.0
    
    s = str(price).strip().replace('$', '').replace(',', '')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
        
    if not s or s.lower() in ("none", "", "nan"):
        return 0.0
        
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _parse_date(date_val) -> Optional[datetime]:
    """Parse Activity Date to datetime."""
    if date_val is None or pd.isna(date_val):
        return None
    try:
        dt = pd.to_datetime(date_val)
        return dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else datetime.combine(dt.date(), datetime.min.time())
    except Exception:
        return None


def parse_robinhood_to_trades(opt_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Pair BTO (Buy To Open) + STC (Sell To Close) / OEXP rows into complete trades.
    Returns a list of dicts ready for Trade model: ticker, option_type, strike, expiry, contracts,
    entry_price, exit_price, entry_time, exit_time, pnl.
    """
    if opt_df.empty:
        return []

    col_map = {c.lower().replace(" ", "_"): c for c in opt_df.columns}
    activity_col = col_map.get("activity_date") or col_map.get("process_date") or (opt_df.columns[0] if len(opt_df.columns) > 0 else None)
    if not activity_col:
        return []
    instrument_col = col_map.get("instrument") or "Instrument"
    desc_col = col_map.get("description") or "Description"
    trans_col = col_map.get("trans_code") or "Trans Code"
    qty_col = col_map.get("quantity") or "Quantity"
    price_col = col_map.get("price") or "Price"
    amount_col = col_map.get("amount") or "Amount"
    
    # Try finding an explicit time column for combinations
    time_col = col_map.get("time") or col_map.get("execution_time") or col_map.get("activity_time")

    # Normalize column names for access
    def _get(row, key, default=None):
        for k, v in col_map.items():
            if k.replace("_", "") == key.replace("_", ""):
                return row.get(v, default)
        return row.get(key, default) if key in opt_df.columns else default

    # Build rows as list of dicts
    rows = []
    # Normalize Robinhood "Buy"/"Sell" to BTO/STC
    trans_map = {"BUY": "BTO", "SELL": "STC"}
    for _, r in opt_df.iterrows():
        row = r.to_dict()
        trans = str(row.get(trans_col, "") or "").strip().upper()
        trans = trans_map.get(trans, trans)
        if trans not in ("BTO", "STC", "OEXP"):
            continue
        instrument = row.get(instrument_col, "")
        parsed = _parse_option_description(str(row.get(desc_col, "")), instrument)
        if not parsed:
            continue
        date_val = row.get(activity_col)
        dt = _parse_date(date_val)
        
        # If there's an explicit time column, merge it into the date
        if time_col and dt and row.get(time_col):
            try:
                t_str = str(row.get(time_col)).strip()
                t_val = pd.to_datetime(t_str).time()
                dt = datetime.combine(dt.date(), t_val)
            except Exception:
                pass
                
        qty = _parse_quantity(row.get(qty_col))
        price = _parse_price(row.get(price_col))
        amount = _parse_price(row.get(amount_col))

        rows.append({
            "trans": trans,
            "ticker": parsed["ticker"],
            "option_type": parsed["option_type"],
            "strike": parsed["strike"],
            "expiry_str": parsed.get("expiry_str", ""),
            "date": dt,
            "contracts": qty,
            "price": price,
            "amount": amount,
            "description": str(row.get(desc_col, "")),
        })

    # Group by (ticker, option_type, strike, expiry_str) and pair BTO with STC/OEXP
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["ticker"], r["option_type"], r["strike"], r["expiry_str"])
        groups[key].append(r)

    trades = []
    
    def safe_localize(dt, default_time_str):
        et = pytz.timezone("America/New_York")
        d = dt.date() if hasattr(dt, "date") else dt
        t = dt.time() if hasattr(dt, "time") and dt.time() != datetime.min.time() else datetime.strptime(default_time_str, "%H:%M").time()
        return et.localize(datetime.combine(d, t)).astimezone(pytz.UTC)

    for key, group in groups.items():
        # Sort chronologically to naturally pair earlier BTOs with earlier STCs
        group_sorted = sorted(group, key=lambda x: x["date"] if x["date"] else datetime.min)
        btos = [x for x in group_sorted if x["trans"] == "BTO"]
        exits = [x for x in group_sorted if x["trans"] in ("STC", "OEXP")]
        
        while btos and exits:
            bto = btos[0]
            ex = exits[0]
            
            contracts = min(bto["contracts"], ex["contracts"])
            if contracts <= 0:
                if bto["contracts"] <= 0: btos.pop(0)
                if ex["contracts"] <= 0: exits.pop(0)
                continue

            entry_price = bto["price"]
            exit_price = ex["price"]
            entry_dt = bto["date"] if bto["date"] else datetime.now()
            exit_dt = ex["date"] if ex["date"] else entry_dt

            safe_entry_dt = safe_localize(entry_dt, "09:35")
            safe_exit_dt = safe_localize(exit_dt, "15:55")

            matched_contracts = contracts
            pnl = (exit_price - entry_price) * 100 * matched_contracts

            # Parse expiry date for Trade.expiry
            expiry_date = None
            if bto.get("expiry_str"):
                try:
                    expiry_date = pd.to_datetime(bto["expiry_str"]).date()
                except Exception:
                    pass
            if not expiry_date and safe_entry_dt:
                expiry_date = safe_entry_dt.date() if hasattr(safe_entry_dt, "date") else None

            ticker = bto["ticker"]
            if ticker.upper() == "SPX":
                ticker = "^SPX"

            trades.append({
                "ticker": ticker,
                "option_type": bto["option_type"],
                "strike": bto["strike"],
                "expiry": expiry_date,
                "contracts": contracts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_time": safe_entry_dt,
                "exit_time": safe_exit_dt,
                "pnl": pnl,
            })
            
            bto["contracts"] -= contracts
            ex["contracts"] -= contracts
            
            if bto["contracts"] <= 0:
                btos.pop(0)
            if ex["contracts"] <= 0:
                exits.pop(0)

    return trades
