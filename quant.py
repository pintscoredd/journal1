import logging
import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq, newton, bisect

MIN_T = 1 / (252 * 390)  # 1 trading minute minimum time floor

logger = logging.getLogger(__name__)

def _d1_d2(S, K, T, r, sigma):
    T = max(T, MIN_T)
    sigma = max(sigma, 1e-8)  # Prevent divide by zero
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, option_type='call'):
    T = max(T, MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    
    if option_type.lower() == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
    return price

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    T = max(T, MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    
    # PDF of d1
    nd1 = stats.norm.pdf(d1)
    
    # Delta
    if option_type.lower() == 'call':
        delta = stats.norm.cdf(d1)
    else:
        delta = stats.norm.cdf(d1) - 1.0
        
    # Gamma (same for both)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    
    # Theta
    # Formula uses T in years, return value is per year, common to divide by 365 for daily
    if option_type.lower() == 'call':
        theta = (- (S * sigma * nd1) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        theta = (- (S * sigma * nd1) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
                 
    # Vega
    vega = S * np.sqrt(T) * nd1
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }

def implied_volatility(price, S, K, T, r, option_type='call'):
    T = max(T, MIN_T)
    
    # Avoid intrinsic issues and deep OTM issues.
    if option_type.lower() == 'call':
        intrinsic = max(0.0, S - K * np.exp(-r * T))
    else:
        intrinsic = max(0.0, K * np.exp(-r * T) - S)
        
    if price < intrinsic:
        return 0.01  # Price is below intrinsic, vol is effectively ~0, clamp to floor
        
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    iv = None
    
    try:
        iv = brentq(objective, 1e-4, 10.0, maxiter=500)
    except Exception as e:
        logger.warning("implied_volatility brentq failed", exc_info=e)
        try:
            def fprime(sigma):
                return bs_greeks(S, K, T, r, sigma, option_type)["vega"]
            iv = newton(objective, x0=0.2, fprime=fprime, maxiter=200)
        except Exception as e2:
            logger.warning("implied_volatility newton failed", exc_info=e2)
            try:
                iv = bisect(objective, 1e-4, 10.0, maxiter=200)
            except Exception as e3:
                logger.error("implied_volatility bisect failed; returning None", exc_info=e3)
                pass
                
    if iv is None or np.isnan(iv):
        return None
        
    return min(max(iv, 0.01), 6.0)

def kelly_fraction(avg_win, avg_loss, win_rate):
    """
    f = (bp - q) / b
    where b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    """
    if avg_loss == 0 or np.isnan(avg_loss):
        return 0.0  # safe guard
    b = abs(avg_win / avg_loss)
    p = win_rate
    q = 1 - p
    
    if b == 0:
        return 0.0
        
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))

def compute_trade_scores(
    theoretical_edge, 
    vol_ratio, 
    delta, 
    gamma_exposure, 
    execution_slippage,
    entry_time_expectancy,
    hold_time_minutes
):
    """
    Computes component scores 0-100 and a weighted final score.
    Weights: 
    volatility_edge_score: 35%
    execution_score: 25%
    timing_score: 25%
    risk_reward_score: 15%
    """
    def _safe_float(val, default=0.0):
        if val is None or np.isnan(val):
            return default
        return float(val)

    vol_ratio = _safe_float(vol_ratio, 1.0)
    execution_slippage = _safe_float(execution_slippage, 0.0)
    entry_time_expectancy = _safe_float(entry_time_expectancy, 0.0)
    gamma_exposure = _safe_float(gamma_exposure, 0.0)

    # 1. Volatility Edge Score (ideal vol_ratio is 1.0)
    # The farther from 1.0, the lower the score.
    # We use a continuous exponential decay function.
    distance = abs(vol_ratio - 1.0)
    vol_score = 100 * np.exp(-1.5 * distance)
    vol_score = min(max(vol_score, 0), 100)

    # 2. Execution Score (scale based on relative slippage to theoretical edge)
    exec_score = 100 * np.exp(-10 * abs(execution_slippage))
    exec_score = min(max(exec_score, 0), 100)
    
    # 3. Timing Score based on expectancies (scale 0 to 1 -> 0 to 100)
    time_score = min(max(100 * entry_time_expectancy, 0), 100)
    
    risk_score = 100 * np.exp(-0.05 * abs(gamma_exposure))
    risk_score = min(max(risk_score, 0), 100)
    
    total = (0.35 * vol_score) + (0.25 * exec_score) + (0.25 * time_score) + (0.15 * risk_score)
    
    return {
        "volatility_edge_score": round(vol_score, 1),
        "execution_score": round(exec_score, 1),
        "timing_score": round(time_score, 1),
        "risk_reward_score": round(risk_score, 1),
        "total_score": round(total, 1)
    }

