"""
Enrich a saved trade with market data, IV, Greeks, and trade quality scores.
Call after saving a trade so that trade_quality_score, implied_vol_*, delta_*, etc. are populated.
"""
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import yfinance as yf

from db import get_session, Trade
from ingest import get_market_data, get_vix_for_day, compute_realized_vol
from quant import (
    implied_volatility,
    bs_greeks,
    bs_price,
    compute_trade_scores,
    MIN_T,
    kelly_fraction,
)

# Risk-free rate proxy (could be fetched from FRED in the future)
DEFAULT_R = 0.05

_logger = logging.getLogger(__name__)
_YF_TICKER_CACHE: dict[str, yf.Ticker] = {}
_OPTION_CHAIN_CACHE: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}


def _years_from_minutes(minutes: float) -> float:
    # 252 trading days, 390 minutes per day
    return max(minutes / (252 * 390), MIN_T)


def _get_underlying_at_time(md: pd.DataFrame, dt: datetime):
    """Return (S, timestamp) for the bar closest to dt. dt and md.index should be timezone-aware."""
    if md.empty:
        return None, None
    target = pd.to_datetime(dt, utc=True)
    if md.index.tz is None:
        md = md.copy()
        md.index = md.index.tz_localize("UTC")
    idx = md.index.get_indexer([target], method="nearest")[0]
    row = md.iloc[idx]
    return float(row["Close"]), md.index[idx]


def _get_yf_ticker(symbol: str) -> yf.Ticker:
    if symbol not in _YF_TICKER_CACHE:
        _YF_TICKER_CACHE[symbol] = yf.Ticker(symbol)
    return _YF_TICKER_CACHE[symbol]


def _get_option_chain(symbol: str, expiry_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = (symbol, expiry_str)
    if key in _OPTION_CHAIN_CACHE:
        return _OPTION_CHAIN_CACHE[key]
    t = _get_yf_ticker(symbol)
    chain = t.option_chain(expiry_str)
    calls = getattr(chain, "calls", pd.DataFrame())
    puts = getattr(chain, "puts", pd.DataFrame())
    _OPTION_CHAIN_CACHE[key] = (calls, puts)
    return calls, puts


def _interp_iv_for_target_delta(
    df: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    option_type: str,
    target_delta: float,
) -> float | None:
    if df.empty or "strike" not in df.columns or "impliedVolatility" not in df.columns:
        return None
    strikes = df["strike"].values.astype(float)
    ivs = df["impliedVolatility"].values.astype(float)
    if not np.isfinite(S) or S <= 0 or len(strikes) == 0:
        return None
    deltas = []
    for k, iv in zip(strikes, ivs):
        if not np.isfinite(iv) or iv <= 0:
            deltas.append(np.nan)
            continue
        try:
            g = bs_greeks(S, float(k), T, r, float(iv), option_type)
            deltas.append(float(g["delta"]))
        except Exception:
            deltas.append(np.nan)
    deltas = np.array(deltas, dtype=float)
    mask = np.isfinite(deltas) & np.isfinite(ivs)
    if not mask.any():
        return None
    strikes = strikes[mask]
    ivs = ivs[mask]
    deltas = deltas[mask]
    if option_type == "put":
        tg = -abs(target_delta)
    else:
        tg = abs(target_delta)
    idx_sorted = np.argsort(deltas)
    deltas_sorted = deltas[idx_sorted]
    ivs_sorted = ivs[idx_sorted]
    if len(deltas_sorted) == 0:
        return None
    if tg <= deltas_sorted[0]:
        return float(ivs_sorted[0])
    if tg >= deltas_sorted[-1]:
        return float(ivs_sorted[-1])
    hi_idx = np.searchsorted(deltas_sorted, tg, side="right")
    lo_idx = hi_idx - 1
    d_lo = deltas_sorted[lo_idx]
    d_hi = deltas_sorted[hi_idx]
    iv_lo = ivs_sorted[lo_idx]
    iv_hi = ivs_sorted[hi_idx]
    if d_hi == d_lo:
        return float(iv_lo)
    w = (tg - d_lo) / (d_hi - d_lo)
    iv = iv_lo + w * (iv_hi - iv_lo)
    if not np.isfinite(iv):
        return None
    return float(iv)


def enrich_trade(trade_id: int) -> str | None:
    """
    Load trade by id, fetch market data, compute IV/Greeks/scores, update and commit.
    Returns None on success, or an error message string.
    """
    session = get_session()
    try:
        trade = session.query(Trade).filter_by(id=trade_id).first()
        if not trade:
            return "Trade not found."

        ticker = trade.ticker or "^SPX"
        md = get_market_data(ticker, "1m")
        if md.empty:
            return "No market data for ticker."

        entry_dt = pd.to_datetime(trade.entry_time)
        exit_dt = pd.to_datetime(trade.exit_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.tz_localize("UTC")
        if exit_dt.tzinfo is None:
            exit_dt = exit_dt.tz_localize("UTC")

        S_entry, _ = _get_underlying_at_time(md, entry_dt)
        S_exit, _ = _get_underlying_at_time(md, exit_dt)
        if S_entry is None:
            return "Could not resolve underlying price at entry."
        if S_exit is None:
            S_exit = S_entry

        trade.underlying_entry_price = S_entry
        trade.underlying_exit_price = S_exit

        # Time to expiry: use hold duration as proxy for 0DTE (remaining life)
        hold_minutes = (exit_dt - entry_dt).total_seconds() / 60.0
        trade.hold_time_minutes = hold_minutes
        T = _years_from_minutes(hold_minutes)

        r = DEFAULT_R
        K = float(trade.strike or 0)
        option_type = (trade.option_type or "call").lower()
        entry_price = float(trade.entry_price or 0)
        exit_price = float(trade.exit_price or 0)

        # VIX and realized vol for vol_ratio
        vix = get_vix_for_day(entry_dt)
        trade.vix_at_entry = vix
        vix_annual = (vix / 100.0) if vix else 0.20
        real_vol_5m = compute_realized_vol(md, entry_dt, 5, "1m")
        real_vol_15m = compute_realized_vol(md, entry_dt, 15, "1m")
        trade.realized_vol_5m = real_vol_5m
        trade.realized_vol_15m = real_vol_15m

        # IV at entry and exit
        iv_entry = implied_volatility(entry_price, S_entry, K, T, r, option_type)
        expiry_date = getattr(trade, "expiry", None)
        expiry_dt = None
        if expiry_date is not None:
            try:
                expiry_dt = datetime.combine(expiry_date, datetime.min.time()).replace(tzinfo=entry_dt.tzinfo)
            except Exception:
                expiry_dt = None
        if expiry_dt and expiry_dt.date() < exit_dt.date():
            _logger.warning(
                "Trade %s has expiry %s before exit %s; clamping exit T to MIN_T and skipping exit IV/Greeks.",
                trade.id,
                expiry_dt.date(),
                exit_dt,
            )
            T_exit = MIN_T
            iv_exit = None
        else:
            # At exit, remaining time is negligible for 0DTE; use 1-min floor for numerical stability
            T_exit = _years_from_minutes(1.0)
            iv_exit = implied_volatility(exit_price, S_exit, K, T_exit, r, option_type) if exit_price else None
        trade.implied_vol_entry = iv_entry
        trade.implied_vol_exit = iv_exit

        if iv_entry is not None:
            greeks_e = bs_greeks(S_entry, K, T, r, iv_entry, option_type)
            trade.delta_entry = greeks_e["delta"]
            trade.gamma_entry = greeks_e["gamma"]
            trade.theta_entry = greeks_e["theta"] / 252.0
            trade.vega_entry = greeks_e["vega"]
            try:
                theoretical_entry = bs_price(S_entry, K, T, r, iv_entry, option_type)
            except Exception:
                theoretical_entry = None
            trade.entry_theoretical_price = theoretical_entry
        else:
            theoretical_entry = None

        if iv_exit is not None:
            greeks_x = bs_greeks(S_exit, K, T, r, iv_exit, option_type)
            trade.delta_exit = greeks_x["delta"]
            trade.gamma_exit = greeks_x["gamma"]
            trade.theta_exit = greeks_x["theta"] / 252.0
            trade.vega_exit = greeks_x["vega"]
            try:
                theoretical_exit = bs_price(S_exit, K, T_exit, r, iv_exit, option_type)
            except Exception:
                theoretical_exit = None
            trade.exit_theoretical_price = theoretical_exit

        ref_vol = vix_annual if vix_annual > 0 else 0.20
        vol_ratio = (iv_entry / ref_vol) if iv_entry and ref_vol else 1.0
        trade.vol_ratio = vol_ratio

        vol_skew = None
        vol_term_slope = None
        try:
            yf_ticker = ticker
            t = _get_yf_ticker(yf_ticker)
            expiries = list(getattr(t, "options", []) or [])
            current_expiry = None
            next_expiry = None
            if expiries:
                entry_date = entry_dt.date()
                for exp_str in expiries:
                    try:
                        exp_date = pd.to_datetime(exp_str).date()
                    except Exception:
                        continue
                    if exp_date >= entry_date:
                        if current_expiry is None:
                            current_expiry = exp_str
                        elif next_expiry is None:
                            next_expiry = exp_str
                            break
                if current_expiry is None:
                    current_expiry = expiries[0]
            if current_expiry:
                calls_curr, puts_curr = _get_option_chain(yf_ticker, current_expiry)
                if expiry_dt is None and expiries:
                    try:
                        expiry_dt = pd.to_datetime(current_expiry).tz_localize(entry_dt.tzinfo)
                    except Exception:
                        expiry_dt = None
                effective_T = _years_from_minutes(60.0)
                if S_entry and S_entry > 0:
                    iv_call_25 = _interp_iv_for_target_delta(
                        calls_curr, S_entry, effective_T, r, "call", 0.25
                    )
                    iv_put_25 = _interp_iv_for_target_delta(
                        puts_curr, S_entry, effective_T, r, "put", 0.25
                    )
                    if iv_call_25 is not None and iv_put_25 is not None:
                        vol_skew = iv_put_25 - iv_call_25
            if current_expiry and next_expiry:
                calls_curr, _ = _get_option_chain(yf_ticker, current_expiry)
                calls_next, _ = _get_option_chain(yf_ticker, next_expiry)
                if (
                    not calls_curr.empty
                    and not calls_next.empty
                    and S_entry
                    and S_entry > 0
                ):
                    strikes_curr = calls_curr["strike"].values.astype(float)
                    ivs_curr = calls_curr["impliedVolatility"].values.astype(float)
                    strikes_next = calls_next["strike"].values.astype(float)
                    ivs_next = calls_next["impliedVolatility"].values.astype(float)
                    atm_target = S_entry
                    idx_curr = int(np.argmin(np.abs(strikes_curr - atm_target)))
                    idx_next = int(np.argmin(np.abs(strikes_next - atm_target)))
                    iv_curr = float(ivs_curr[idx_curr]) if 0 <= idx_curr < len(ivs_curr) else None
                    iv_next = float(ivs_next[idx_next]) if 0 <= idx_next < len(ivs_next) else None
                    if iv_curr is not None and iv_next is not None:
                        vol_term_slope = iv_next - iv_curr
        except Exception:
            vol_skew = None
            vol_term_slope = None

        trade.vol_skew = vol_skew
        trade.vol_term_slope = vol_term_slope

        if theoretical_entry is not None and theoretical_entry > 0:
            execution_slippage = abs(entry_price - theoretical_entry) / max(theoretical_entry, 0.01)
        else:
            execution_slippage = 0.0
        pst_time = entry_dt.tz_convert("America/Los_Angeles")
        entry_time_expectancy = 0.5
        try:
            session_hist = get_session()
            try:
                history = (
                    session_hist.query(Trade)
                    .filter(Trade.entry_time < trade.entry_time)
                    .order_by(Trade.entry_time.asc())
                    .all()
                )
            finally:
                session_hist.close()
            if history and len(history) >= 20:
                hist_rows = []
                for h in history:
                    d = h.__dict__.copy()
                    d.pop("_sa_instance_state", None)
                    hist_rows.append(d)
                hist_df = pd.DataFrame(hist_rows)
                if not hist_df.empty and "pnl" in hist_df.columns and "entry_time" in hist_df.columns:
                    hist_df = hist_df.dropna(subset=["pnl", "entry_time"])
                    if not hist_df.empty:
                        hist_df["entry_time"] = pd.to_datetime(hist_df["entry_time"])
                        if hist_df["entry_time"].dt.tz is None:
                            hist_df["entry_time"] = hist_df["entry_time"].dt.tz_localize("UTC")
                        hist_df["entry_time_pst"] = hist_df["entry_time"].dt.tz_convert(
                            "America/Los_Angeles"
                        )
                        hist_df["time_bucket"] = hist_df["entry_time_pst"].dt.floor("15T").dt.time
                        bucket_means = (
                            hist_df.groupby("time_bucket")["pnl"].mean().dropna()
                        )
                        if not bucket_means.empty:
                            current_bucket = pst_time.floor("15T").time()
                            if current_bucket in bucket_means.index:
                                bucket_val = float(bucket_means.loc[current_bucket])
                                min_exp = float(bucket_means.min())
                                max_exp = float(bucket_means.max())
                                if max_exp > min_exp:
                                    norm = (bucket_val - min_exp) / (max_exp - min_exp)
                                else:
                                    norm = 0.5
                                entry_time_expectancy = float(
                                    np.clip(norm, 0.0, 1.0)
                                )
                        else:
                            entry_time_expectancy = 0.5
        except Exception:
            entry_time_expectancy = 0.5

        gamma_exposure = abs((trade.gamma_entry or 0) * (trade.contracts or 1) * 100)
        scores = compute_trade_scores(
            theoretical_edge=0.05,
            vol_ratio=vol_ratio,
            delta=trade.delta_entry or 0.5,
            gamma_exposure=gamma_exposure,
            execution_slippage=execution_slippage,
            entry_time_expectancy=entry_time_expectancy,
            hold_time_minutes=hold_minutes,
        )
        trade.entry_execution_score = scores["execution_score"]
        trade.volatility_edge_score = scores["volatility_edge_score"]
        trade.timing_score = scores["timing_score"]
        trade.risk_reward_score = scores["risk_reward_score"]
        trade.trade_quality_score = scores["total_score"]

        mae = None
        try:
            mask_path = (md.index >= entry_dt) & (md.index <= exit_dt)
            window = md.loc[mask_path].copy()
            if not window.empty and iv_entry is not None:
                closes = window["Close"].astype(float).values
                idx = window.index
                elapsed_minutes = (idx - entry_dt).total_seconds() / 60.0
                remaining_minutes = np.maximum(hold_minutes - elapsed_minutes, 1.0)
                T_path = _years_from_minutes(remaining_minutes)
                try:
                    opt_prices = np.array(
                        [
                            bs_price(
                                float(S_t),
                                K,
                                float(T_t),
                                r,
                                float(iv_entry),
                                option_type,
                            )
                            for S_t, T_t in zip(closes, T_path)
                        ],
                        dtype=float,
                    )
                except Exception:
                    opt_prices = np.array([], dtype=float)
                if opt_prices.size > 0:
                    contracts = float(trade.contracts or 1)
                    pnl_path = (opt_prices - entry_price) * 100.0 * contracts
                    mae = float(np.min(pnl_path))
        except Exception:
            mae = None

        if mae is not None and mae < 0:
            trade.max_loss = mae
            denom = abs(mae)
            if denom > 0:
                trade.r_multiple = float((trade.pnl or 0.0) / denom)

        N_hist = 20
        try:
            session_hist2 = get_session()
            try:
                prev_trades = (
                    session_hist2.query(Trade)
                    .filter(Trade.entry_time < trade.entry_time)
                    .order_by(Trade.entry_time.desc())
                    .limit(N_hist)
                    .all()
                )
            finally:
                session_hist2.close()
            if prev_trades:
                pnls = [float(t.pnl or 0.0) for t in prev_trades]
                pnls_arr = np.array(pnls, dtype=float)
                if pnls_arr.size > 0 and np.any(pnls_arr != 0):
                    wins = pnls_arr[pnls_arr > 0]
                    losses = pnls_arr[pnls_arr < 0]
                    num_trades = pnls_arr.size
                    num_wins = wins.size
                    win_rate = num_wins / num_trades if num_trades > 0 else 0.0
                    avg_win = float(wins.mean()) if wins.size > 0 else 0.0
                    avg_loss = float(np.abs(losses.mean())) if losses.size > 0 else 0.0
                    f = kelly_fraction(avg_win=avg_win, avg_loss=avg_loss, win_rate=win_rate)
                    base_capital = 3000.0
                    allocation = base_capital * float(f)
                    per_contract_risk = abs((trade.entry_price or 0.0) * 100.0)
                    if per_contract_risk > 0:
                        contracts = int(max(1, allocation // per_contract_risk))
                    else:
                        contracts = 1
                    trade.recommended_contracts = contracts
                else:
                    trade.recommended_contracts = 1
            else:
                trade.recommended_contracts = 1
        except Exception:
            trade.recommended_contracts = trade.recommended_contracts or 1

        session.commit()
        return None
    except Exception as e:
        session.rollback()
        return str(e)
    finally:
        session.close()
