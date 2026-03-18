import json
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import pytz
from datetime import datetime, timedelta

from db import get_session, Trade, Secret
from secrets_store import store_api_key, get_master_key, encrypt_key
from ingest import get_market_data, import_trades_csv, parse_robinhood_to_trades, filter_option_trades
from ai_adapter import AIAdapter
from montecarlo import simulate_equity_paths, calculate_risk_metrics
from enrichment import enrich_trade
from utils import get_local_today, to_local_time

# --- CACHING ---
@st.cache_data(ttl=3600)
def fetch_cached_market_data(ticker, interval, start=None, end=None):
    return get_market_data(ticker, interval, start, end)

@st.cache_data
def run_cached_monte_carlo(trades_df, num_sims, initial, block_size):
    if trades_df.empty:
        return None, None
    paths = simulate_equity_paths(trades_df, num_sims, initial, block_size)
    metrics = calculate_risk_metrics(paths)
    return paths, metrics

# --- DATA FETCH ---
def parse_time(ts: str) -> datetime.time:
    ts = ''.join(filter(str.isdigit, ts))
    if not ts: return datetime.strptime("070000", "%H%M%S").time()
    if len(ts) <= 2: ts = ts.zfill(2) + "0000"
    elif len(ts) == 3: ts = "0" + ts + "00"
    elif len(ts) == 4: ts = ts + "00"
    elif len(ts) == 5: ts = "0" + ts
    elif len(ts) > 6: ts = ts[:6]
    try:
        return datetime.strptime(ts, "%H%M%S").time()
    except:
        return datetime.strptime("070000", "%H%M%S").time()

def get_all_trades_df():
    session = get_session()
    try:
        trades = session.query(Trade).order_by(Trade.entry_time.asc()).all()
        if not trades:
            return pd.DataFrame()
        # Convert to records
        data = []
        for t in trades:
            d = t.__dict__.copy()
            d.pop('_sa_instance_state', None)
            data.append(d)
        df = pd.DataFrame(data)
        if not df.empty:
            try:
                # Persistent backup to prevent data loss on container/app reboot
                import os
                backup_dir = os.path.expanduser('~/.spx_0dte_journal')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, "trades_backup.csv")
                df.to_csv(backup_path, index=False)
            except Exception:
                pass
        return df
    finally:
        session.close()

# --- HELPERS ---
def _filter_option_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper over ingest.filter_option_trades for backward compatibility."""
    return filter_option_trades(df)

# --- PAGES ---
def render_dashboard():
    st.header("Dashboard")
    df = get_all_trades_df()
    if df.empty:
        st.info("No trades found. Please import some trades.")
        return

    df = df.sort_values('entry_time')
    total_pnl = df['pnl'].sum()
    win_rate = len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if len(df[df['pnl'] <= 0]) > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PnL", f"${total_pnl:.2f}")
    c2.metric("Win Rate", f"{win_rate*100:.1f}%")
    c3.metric("Avg Win", f"${avg_win:.2f}")
    c4.metric("Avg Loss", f"${avg_loss:.2f}")

    df = df.sort_values('entry_time')
    df['cum_pnl'] = df['pnl'].cumsum()
    df['Trade Number'] = range(1, len(df) + 1)
    
    # Equity Curve
    fig = px.line(df, x='Trade Number', y='cum_pnl', title='Equity Curve', markers=True,
                  hover_data=['entry_time', 'ticker', 'pnl'])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font=dict(color="#d3d3d3", family="Fira Code, monospace"))
    fig.update_traces(line_color="#d3d3d3", marker=dict(color="#d3d3d3", size=8))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Expectancy by 15m bucket
    st.subheader("Rolling Expectancy by Time Of Day (15m Bins)")
    # Convert entry times to PST before bucketing
    df_pst_times = pd.to_datetime(df['entry_time']).dt.tz_convert('America/Los_Angeles')
    df['time_bucket'] = df_pst_times.dt.floor('15T').dt.time
    expectancy_df = df.groupby('time_bucket')['pnl'].mean().reset_index()
    # convert time_bucket to string for plotting
    expectancy_df['time_bucket_str'] = expectancy_df['time_bucket'].apply(lambda x: x.strftime("%H:%M"))
    fig2 = px.bar(expectancy_df, x='time_bucket_str', y='pnl', title='Average PnL by 15m Entry Window')
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font=dict(color="#d3d3d3", family="Fira Code, monospace"))
    fig2.update_traces(marker_color="#d3d3d3")
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Monte Carlo Sample
    st.subheader("Monte Carlo Simulation (Block Bootstrap)")
    num_sims = st.session_state.get('mc_sims', 1000)
    paths, metrics = run_cached_monte_carlo(df, num_sims, initial=st.session_state.get('capital', 200), block_size=3)
    if paths is not None:
        cr1, cr2 = st.columns(2)
        cr1.metric("Probability of Ruin", f"{metrics['probability_of_ruin']*100:.1f}%")
        cr2.metric("Expected Worst Drawdown", f"${metrics['expected_worst_drawdown']:.2f}")
        
        # Plot a subset of paths
        plot_paths = paths[:, :min(50, num_sims)]
        fig_mc = go.Figure()
        for i in range(plot_paths.shape[1]):
            fig_mc.add_trace(go.Scatter(y=plot_paths[:, i], mode='lines', line=dict(width=1, color='rgba(211, 211, 211, 0.2)'), showlegend=False))
        fig_mc.update_layout(title=f"Sample 50 Equity Paths (from {num_sims})", template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font=dict(color="#d3d3d3", family="Fira Code, monospace"))
        fig_mc.update_xaxes(showgrid=False)
        fig_mc.update_yaxes(showgrid=False)
        st.plotly_chart(fig_mc, use_container_width=True)

def render_new_trade():
    st.header("Ingest Trades")
    
    file = st.file_uploader("Upload CSV (Robinhood format example)", type=["csv"])
    if file:
        try:
            raw_df = import_trades_csv(file)
        except ValueError as e:
            st.error(str(e))
            raw_df = None

        if raw_df is not None:
            opt_df = _filter_option_trades(raw_df)

            if opt_df is raw_df:
                st.info("No clear option rows detected; showing full CSV.")
            else:
                st.subheader("Detected option trades from CSV")

            # Parse into paired trades (BTO + STC)
            paired_trades = parse_robinhood_to_trades(opt_df)

            if not paired_trades:
                st.warning("No complete option trades (BTO + STC pairs) found. Unpaired rows are skipped.")
                # Fallback: show raw table for reference
                with st.expander("View raw option rows"):
                    styled_df = opt_df.style.set_properties(**{'background-color': '#050505', 'color': '#d3d3d3', 'border-color': '#555555'})
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Review each trade and toggle **Include** to approve or deny before importing.")

                # Card-based layout: 2 cards per row
                cols_per_row = 2
                for i in range(0, len(paired_trades), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, col in enumerate(row_cols):
                        idx = i + j
                        if idx >= len(paired_trades):
                            break
                        t = paired_trades[idx]
                        with col:
                            pnl = t.get("pnl", 0)
                            pnl_color = "#26a69a" if pnl >= 0 else "#ef5350"
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(135deg, #1e1e2e 0%, #262730 100%);
                                    border-radius: 12px;
                                    padding: 1rem 1.2rem;
                                    margin-bottom: 1rem;
                                    border-left: 4px solid {pnl_color};
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                                ">
                                    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem;">
                                        {t.get('ticker', '')} {t.get('option_type', '').capitalize()} ${t.get('strike', 0):.2f}
                                    </div>
                                    <div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">
                                        {str(t.get('expiry', '')) if t.get('expiry') else '—'} · {t.get('contracts', 1)} contract(s)
                                    </div>
                                    <div style="font-size: 0.9rem;">
                                        ${t.get('entry_price', 0):.2f} → ${t.get('exit_price', 0):.2f}
                                        <span style="color: {pnl_color}; font-weight: 600; margin-left: 0.5rem;">
                                            {f'+${pnl:.0f}' if pnl >= 0 else f'-${abs(pnl):.0f}'}
                                        </span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Time inputs with defaults from parser
                            en_str_def = pd.to_datetime(t.get('entry_time')).tz_convert('America/Los_Angeles').strftime('%H:%M:%S')
                            ex_str_def = pd.to_datetime(t.get('exit_time')).tz_convert('America/Los_Angeles').strftime('%H:%M:%S')
                            
                            tc1, tc2 = st.columns(2)
                            with tc1:
                                st.text_input("Entry Time (PST)", value=en_str_def, key=f"csv_en_{idx}")
                            with tc2:
                                st.text_input("Exit Time (PST)", value=ex_str_def, key=f"csv_ex_{idx}")
                                
                            include = st.checkbox("Include", value=True, key=f"csv_approve_{idx}")

                approved_count = sum(1 for i in range(len(paired_trades)) if st.session_state.get(f"csv_approve_{i}", True))
                denied_count = len(paired_trades) - approved_count
                st.write(f"**{approved_count}** approved · **{denied_count}** denied")

                if st.button("Import approved trades"):
                    import uuid
                    session = get_session()
                    saved = 0
                    imported_ids = []
                    try:
                        for i, t in enumerate(paired_trades):
                            if not st.session_state.get(f"csv_approve_{i}", True):
                                continue
                            expiry = t.get("expiry")
                            if hasattr(expiry, "date"):
                                expiry = expiry.date() if expiry else None
                            elif expiry is not None and not hasattr(expiry, "year"):
                                expiry = pd.to_datetime(expiry).date() if pd.notna(expiry) else None
                                
                            # Retrieve edited time inputs from state
                            en_str = st.session_state.get(f"csv_en_{i}", pd.to_datetime(t.get('entry_time')).tz_convert('America/Los_Angeles').strftime('%H:%M:%S'))
                            ex_str = st.session_state.get(f"csv_ex_{i}", pd.to_datetime(t.get('exit_time')).tz_convert('America/Los_Angeles').strftime('%H:%M:%S'))
                            
                            # Parse custom time
                            p_en = parse_time(en_str)
                            p_ex = parse_time(ex_str)
                            
                            # Combine with date and convert to UTC
                            orig_en_date = pd.to_datetime(t.get("entry_time")).tz_convert('America/Los_Angeles').date()
                            orig_ex_date = pd.to_datetime(t.get("exit_time")).tz_convert('America/Los_Angeles').date()
                            
                            dt_en = datetime.combine(orig_en_date, p_en)
                            dt_ex = datetime.combine(orig_ex_date, p_ex)
                            
                            utc_en = pd.to_datetime(dt_en).tz_localize('America/Los_Angeles').tz_convert('UTC')
                            utc_ex = pd.to_datetime(dt_ex).tz_localize('America/Los_Angeles').tz_convert('UTC')

                            new_trade = Trade(
                                trade_uuid=str(uuid.uuid4()),
                                ticker=str(t.get("ticker", "")),
                                option_type=str(t.get("option_type", "call")),
                                strike=float(t.get("strike", 0)),
                                expiry=expiry,
                                contracts=int(t.get("contracts", 1)),
                                entry_price=float(t.get("entry_price", 0)),
                                exit_price=float(t.get("exit_price", 0)),
                                entry_time=utc_en,
                                exit_time=utc_ex,
                                pnl=float(t.get("pnl", 0)),
                            )
                            session.add(new_trade)
                            session.flush()
                            imported_ids.append(new_trade.id)
                            saved += 1
                        session.commit()
                        with st.spinner("Computing quant metrics..."):
                            for tid in imported_ids:
                                enrich_trade(tid)
                        try:
                            run_cached_monte_carlo.clear()
                        except Exception:
                            pass
                        st.success(f"Imported **{saved}** trades into your journal.")
                        st.rerun()
                    except Exception as e:
                        session.rollback()
                        st.error(f"Import failed: {e}")
                    finally:
                        session.close()
    
    st.subheader("Manual Quick Entry")
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        ticker = col1.text_input("Ticker", "^SPX")
        option_type = col2.selectbox("Option Type", ["call", "put"])
        
        col3, col4 = st.columns(2)
        strike = col3.number_input("Strike", value=5000.0, step=5.0)
        contracts = col4.number_input("Contracts", min_value=1, value=1, step=1)
        
        c1, c2 = st.columns(2)
        entry_price = c1.number_input("Entry Price", value=5.0, step=0.1)
        exit_price = c2.number_input("Exit Price", value=6.0, step=0.1)
        
        t1, t2, t3 = st.columns(3)
        trade_date = t1.date_input("Trade Date", value=get_local_today())
        entry_time_input = t2.text_input("Entry Time (PST) e.g. 07:15:30", value="07:00:00")
        exit_time_input = t3.text_input("Exit Time (PST) e.g. 07:30:15", value="07:15:00")
        
        submitted = st.form_submit_button("Save Trade")
        if submitted:
            import uuid
            session = get_session()
            try:
                
                parsed_entry = parse_time(entry_time_input)
                parsed_exit = parse_time(exit_time_input)
                # Combine date and time
                entry_dt = datetime.combine(trade_date, parsed_entry)
                exit_dt = datetime.combine(trade_date, parsed_exit)
                
                # Ensure UTC awareness for yfinance lookup
                entry_dt_utc = pd.to_datetime(entry_dt).tz_localize('America/Los_Angeles').tz_convert('UTC')
                exit_dt_utc = pd.to_datetime(exit_dt).tz_localize('America/Los_Angeles').tz_convert('UTC')
                
                # Future data check
                if entry_dt_utc > datetime.now(pytz.utc) + timedelta(minutes=5):
                    st.warning("⚠️ Trade date/time appears to be in the future. yfinance will not have data for this yet.")
                
                new_trade = Trade(
                    trade_uuid=str(uuid.uuid4()),
                    ticker=ticker,
                    option_type=option_type,
                    strike=strike,
                    expiry=trade_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=entry_dt_utc,
                    exit_time=exit_dt_utc,
                    contracts=contracts,
                    pnl=(exit_price - entry_price) * 100 * contracts
                )
                session.add(new_trade)
                session.flush()
                tid = new_trade.id
                session.commit()
                with st.spinner("Computing quant metrics..."):
                    err = enrich_trade(tid)
                if err:
                    st.warning(f"Trade saved but quant computation failed: {err}")
                else:
                    st.success(f"Trade for {ticker} successfully saved: ${new_trade.pnl:.2f} PnL and quant metrics computed!")
            except Exception as e:
                st.error(f"Error saving trade: {e}")
            finally:
                session.close()

def render_trade_viewer():
    st.header("Trade Viewer & Replay")
    df = get_all_trades_df()
    if df.empty:
        st.info("No trades to display.")
        return
        
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
        
    def format_trade(x):
        row = df[df['id'] == x].iloc[0]
        ticker = row.get('ticker', '').replace('^', '')
        op_type = str(row.get('option_type', '')).upper()
        strike = row.get('strike', '')
        try:
            # We want to display the time that was saved.
            # Entry_time is stored in UTC, so we convert it to the local display timezone (PST).
            db_time = pd.to_datetime(row['entry_time'])
            if db_time.tzinfo is None:
                db_time = pytz.utc.localize(db_time)
            etime = db_time.astimezone(pytz.timezone("America/Los_Angeles")).strftime('%b %d, %Y %I:%M:%S %p %Z')
        except:
            etime = row['entry_time']
        pnl = row.get('pnl', 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        return f"Trade {x} | {ticker} ${strike} {op_type} | {etime} | {pnl_str}"

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        trade_ids = df.sort_values('entry_time', ascending=False)['id'].values
        trade_id = st.selectbox("Select Trade", trade_ids, format_func=format_trade)
    with col2:
        st.write("") # push down to align
        st.write("")
        if st.button("Edit Trade", use_container_width=True):
            st.session_state.edit_mode = not st.session_state.edit_mode
    with col3:
        st.write("") # push down to align
        st.write("")
        if st.button("Delete Trade", type="primary", use_container_width=True):
            session = get_session()
            try:
                t_del = session.query(Trade).filter_by(id=trade_id).first()
                if t_del:
                    session.delete(t_del)
                    session.commit()
                    try:
                        run_cached_monte_carlo.clear()
                    except Exception:
                        pass
                    st.rerun()
            except Exception as e:
                st.error("Error deleting.")
            finally:
                session.close()

    selected = df[df['id'] == trade_id].iloc[0]

    if st.button("Recompute quant metrics (IV, Greeks, score)"):
        err = enrich_trade(int(trade_id))
        if err:
            st.error(err)
        else:
            st.success("Trade enriched. Refresh the viewer.")
            st.rerun()
    
    st.write(f"**Score**: {selected.get('trade_quality_score', 'N/A')}/100")
    st.caption(
        f"**Breakdown** — Vol Edge: {selected.get('volatility_edge_score', 'N/A')} | "
        f"Execution: {selected.get('entry_execution_score', 'N/A')} | "
        f"Timing: {selected.get('timing_score', 'N/A')} | "
        f"Risk/Reward: {selected.get('risk_reward_score', 'N/A')}"
    )
    st.caption(
        f"**Vol Surface** — Skew(25Δ): {selected.get('vol_skew', 'N/A')} | "
        f"Term Slope: {selected.get('vol_term_slope', 'N/A')}"
    )
    pnl_val = selected['pnl']
    pnl_color = "#4af626" if pnl_val >= 0 else "#ff3333"
    st.markdown(f"<span style='font-size:1.1rem;'>PnL: <b style='color:{pnl_color}'>${pnl_val:.2f}</b></span>", unsafe_allow_html=True)
    mae_val = selected.get("max_loss")
    r_mult = selected.get("r_multiple")
    rec_contracts = selected.get("recommended_contracts")
    st.caption(
        f"**Path Risk** — MAE: {('N/A' if mae_val is None else f'${mae_val:.2f}')} | "
        f"R-multiple: {('N/A' if r_mult is None else f'{r_mult:.2f}')} | "
        f"Kelly Recommended Contracts: {('N/A' if rec_contracts is None else int(rec_contracts))}"
    )
    
    if st.session_state.edit_mode:
        st.markdown("### Edit Selected Trade")
        with st.form(f"edit_trade_{trade_id}"):
            db_en = pd.to_datetime(selected['entry_time'])
            db_ex = pd.to_datetime(selected['exit_time'])
            if db_en.tzinfo is None: db_en = db_en.tz_localize('UTC')
            if db_ex.tzinfo is None: db_ex = db_ex.tz_localize('UTC')
            
            en_pst = db_en.tz_convert('America/Los_Angeles')
            ex_pst = db_ex.tz_convert('America/Los_Angeles')
            
            e_col1, e_col2 = st.columns(2)
            n_tick = e_col1.text_input("Ticker", value=str(selected['ticker']))
            n_opt = e_col2.selectbox("Option Type", ["call", "put"], index=0 if selected['option_type']=='call' else 1)
            
            e_col3, e_col4, e_col5 = st.columns(3)
            n_strike = e_col3.number_input("Strike", value=float(selected['strike'] or 0))
            n_contracts = e_col4.number_input("Contracts", value=int(selected.get('contracts') or 1), step=1)
            
            e_col6, e_col7 = st.columns(2)
            n_en = e_col6.number_input("Entry Price", value=float(selected['entry_price'] or 0), step=0.1)
            n_ex = e_col7.number_input("Exit Price", value=float(selected['exit_price'] or 0), step=0.1)
            
            t_col1, t_col2, t_col3 = st.columns(3)
            n_date = t_col1.date_input("Trade Date", value=en_pst.date())
            n_t_en = t_col2.text_input("Entry Time (PST)", value=en_pst.strftime("%H:%M:%S"))
            n_t_ex = t_col3.text_input("Exit Time (PST)", value=ex_pst.strftime("%H:%M:%S"))
            
            if st.form_submit_button("Save Changes"):
                sess = get_session()
                try:
                    t_upd = sess.query(Trade).filter_by(id=trade_id).first()
                    if t_upd:
                        p_en = parse_time(n_t_en)
                        p_ex = parse_time(n_t_ex)
                        
                        dt_en = datetime.combine(n_date, p_en)
                        dt_ex = datetime.combine(n_date, p_ex)
                        n_utc_en = pd.to_datetime(dt_en).tz_localize('America/Los_Angeles').tz_convert('UTC')
                        n_utc_ex = pd.to_datetime(dt_ex).tz_localize('America/Los_Angeles').tz_convert('UTC')
                        
                        t_upd.ticker = n_tick
                        t_upd.option_type = n_opt
                        t_upd.strike = n_strike
                        t_upd.expiry = n_date
                        t_upd.contracts = n_contracts
                        t_upd.entry_price = n_en
                        t_upd.exit_price = n_ex
                        t_upd.entry_time = n_utc_en
                        t_upd.exit_time = n_utc_ex
                        t_upd.pnl = (n_ex - n_en) * 100 * n_contracts
                        sess.commit()
                        st.session_state.edit_mode = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")
                finally:
                    sess.close()
    
    entry_context = {}
    chart_data = None
    ema5_data = None
    img = None
    
    # Replay Chart
    st.subheader("Replay Chart")
    try:
        # Parse db time safely first so we can determine age
        db_entry = pd.to_datetime(selected['entry_time'])
        db_exit = pd.to_datetime(selected['exit_time'])
        if db_entry.tzinfo is None:
            db_entry = db_entry.tz_localize('UTC')
        if db_exit.tzinfo is None:
            db_exit = db_exit.tz_localize('UTC')
            
        now = pd.Timestamp.utcnow()
        days_ago = (now - db_entry).days
        
        interval = "1m"
        if days_ago > 720:
            interval = "1d"
        elif days_ago > 50:
            interval = "1h"
        elif days_ago > 5:
            interval = "5m"
            
        # yf.history with 1m/5m interval is very sensitive. 
        # We fetch from 1 day before entry to 1 day after exit to ensure the bars are caught.
        start_date_str = (db_entry - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date_str = (db_exit + timedelta(days=1)).strftime('%Y-%m-%d')
        
        md = fetch_cached_market_data(selected['ticker'], interval, start=start_date_str, end=end_date_str)
        if not md.empty:
            # Expand the bounding window to ensure we catch enough data even if times are slightly off
            # For 1m/5m data, we look for data +/- 5 hours around the trade
            start_date = db_entry - timedelta(hours=5)
            end_date = db_exit + timedelta(hours=5)
            mask = (md.index >= start_date) & (md.index <= end_date)
            plot_md = md.loc[mask].copy()
            plot_md = plot_md.sort_index()
            plot_md = plot_md[~plot_md.index.duplicated(keep='first')]
            
            if not plot_md.empty:
                # Calculate VWAP
                plot_md['typ'] = (plot_md['High'] + plot_md['Low'] + plot_md['Close']) / 3
                plot_md['vwap'] = (plot_md['typ'] * plot_md['Volume']).cumsum() / plot_md['Volume'].cumsum()
                plot_md['ema5'] = plot_md['Close'].ewm(span=5, adjust=False).mean()
                plot_md['ema14'] = plot_md['Close'].ewm(span=14, adjust=False).mean()
                plot_md['ema25'] = plot_md['Close'].ewm(span=25, adjust=False).mean()
                
                closest_idx = plot_md.index.get_indexer([db_entry], method='nearest')[0]
                entry_row = plot_md.iloc[closest_idx]
                
                pre_candles = []
                # grab the 3 precedence candles preceding entry
                for i in range(1, 4):
                    if closest_idx - i >= 0:
                        row = plot_md.iloc[closest_idx - i]
                        pre_candles.append({
                            "offset_minutes": -i,
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "ema5": float(row["ema5"]),
                            "ema14": float(row["ema14"]),
                            "ema25": float(row["ema25"])
                        })
                
                entry_context = {
                    "underlying_price_at_entry": float(entry_row["Close"]),
                    "ema5_at_entry": float(entry_row["ema5"]),
                    "ema14_at_entry": float(entry_row["ema14"]),
                    "ema25_at_entry": float(entry_row["ema25"]),
                    "preceding_3_candles": pre_candles,
                }
                
                chart_data = []
                for idx, row in plot_md.iterrows():
                    # For Lightweight Charts to show PST on the axis, we pass the local (PST) timestamp
                    local_ts = to_local_time(idx).timestamp()
                    chart_data.append({
                        "time": int(local_ts),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"])
                    })
                
                vwap_data = [{"time": int(to_local_time(idx).timestamp()), "value": float(val)} for idx, val in plot_md["vwap"].dropna().items()]
                ema5_data = [{"time": int(to_local_time(idx).timestamp()), "value": float(val)} for idx, val in plot_md["ema5"].dropna().items()]
                ema14_data = [{"time": int(to_local_time(idx).timestamp()), "value": float(val)} for idx, val in plot_md["ema14"].dropna().items()]
                ema25_data = [{"time": int(to_local_time(idx).timestamp()), "value": float(val)} for idx, val in plot_md["ema25"].dropna().items()]
                
                entry_pst_ts = int(to_local_time(db_entry).timestamp())
                exit_pst_ts = int(to_local_time(db_exit).timestamp())

                # Markers need strictly to be mapped to data domain correctly.
                # Find closest timestamps in local PST data to avoid chart errors
                closest_entry = min(chart_data, key=lambda x: abs(x['time'] - entry_pst_ts))['time']
                closest_exit = min(chart_data, key=lambda x: abs(x['time'] - exit_pst_ts))['time']
                
                entry_ts = closest_entry
                exit_ts = closest_exit
                
                chart_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
                </head>
                <body style="margin: 0; background-color: #0e1117;">
                    <div id="chart"></div>
                    <script>
                        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                            width: window.innerWidth,
                            height: 400,
                            layout: {{
                                backgroundColor: '#0e1117',
                                textColor: '#d1d4dc',
                            }},
                            grid: {{
                                vertLines: {{ visible: false }},
                                horzLines: {{ visible: false }},
                            }},
                            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                            timeScale: {{ timeVisible: true, secondsVisible: false }},
                        }});
                        
                        const candlestickSeries = chart.addCandlestickSeries({{
                            upColor: '#26a69a',
                            downColor: '#ef5350',
                            borderVisible: false,
                            wickUpColor: '#26a69a',
                            wickDownColor: '#ef5350'
                        }});
                        const data = {json.dumps(chart_data)};
                        candlestickSeries.setData(data);
                        
                        const vwapSeries = chart.addLineSeries({{
                            color: '#eade52',
                            lineWidth: 2,
                            title: 'VWAP'
                        }});
                        vwapSeries.setData({json.dumps(vwap_data)});

                        chart.addLineSeries({{ color: '#2962FF', lineWidth: 1, title: 'EMA 5'}}).setData({json.dumps(ema5_data)});
                        chart.addLineSeries({{ color: '#FF6D00', lineWidth: 1, title: 'EMA 14'}}).setData({json.dumps(ema14_data)});
                        chart.addLineSeries({{ color: '#00C853', lineWidth: 1, title: 'EMA 25'}}).setData({json.dumps(ema25_data)});
                        
                        let markers = [
                            {{ time: {entry_ts}, position: 'belowBar', color: '#26a69a', shape: 'arrowUp', text: 'Entry' }},
                            {{ time: {exit_ts}, position: 'aboveBar', color: '#ef5350', shape: 'arrowDown', text: 'Exit' }}
                        ];
                        // lightweight-charts requires markers to be perfectly strictly sorted
                        markers.sort((a, b) => a.time - b.time);
                        candlestickSeries.setMarkers(markers);
                        chart.timeScale().fitContent();
                        
                        window.addEventListener('resize', () => {{
                            chart.resize(window.innerWidth, 400);
                        }});
                    </script>
                </body>
                </html>
                """
                components.html(chart_html, height=400)
            else:
                st.warning(
                    f"No market data bounded between {to_local_time(start_date).strftime('%Y-%m-%d %H:%M')} "
                    f"and {to_local_time(end_date).strftime('%Y-%m-%d %H:%M')} (PST)."
                )
                st.info("Tip: Double check if your entry time was during regular market hours! If you just saved this trade after 5 PM PST, ensure the date is set to today.")
    except Exception as e:
        st.warning(f"Failed to load replay: {e}")

    # AI Critique
    if st.button("Generate AI Critique"):
        provider = st.session_state.get('ai_provider', 'gemini')
        try:
            adapter = AIAdapter(provider=provider)
            with open("single_trade_critique.txt", "r") as f:
                template = f.read()
            
            # Serialize quant + hold time and vol context for accurate AI critique
            quant_dict = {
                "ticker": str(selected.get("ticker", "UNKNOWN")),
                "option_type": str(selected.get("option_type", "UNKNOWN")),
                "strike": float(selected.get("strike") or 0.0),
                "entry_price": float(selected.get("entry_price") or 0.0),
                "exit_price": float(selected.get("exit_price") or 0.0),
                "contracts": int(selected.get("contracts") or 1),
                "pnl": float(selected.get("pnl") or 0.0),
                "hold_minutes": float(selected.get("hold_time_minutes") or 15.0),
                "underlying_price_at_entry": entry_context.get("underlying_price_at_entry", 0.0),
                "ema5_at_entry": entry_context.get("ema5_at_entry", 0.0),
                "ema14_at_entry": entry_context.get("ema14_at_entry", 0.0),
                "ema25_at_entry": entry_context.get("ema25_at_entry", 0.0),
                "preceding_3_candles": entry_context.get("preceding_3_candles", []),
                "delta": float(selected.get("delta_entry") or 0.45),
                "gamma_exposure": float(selected.get("gamma_entry") or 0.08),
                "vol_ratio": float(selected.get("vol_ratio") or 1.15),
                "implied_vol_entry": float(selected.get("implied_vol_entry") or 0.0),
                "vix_at_entry": float(selected.get("vix_at_entry") or 0.0),
                "vol_skew_25d": float(selected.get("vol_skew") or 0.0),
                "vol_term_slope": float(selected.get("vol_term_slope") or 0.0),
                "max_adverse_excursion": float(selected.get("max_loss") or 0.0),
                "r_multiple": float(selected.get("r_multiple") or 0.0),
                "recommended_contracts": int(selected.get("recommended_contracts") or 0),
                "execution_score": float(selected.get("entry_execution_score") or 85.0),
                "volatility_edge_score": float(selected.get("volatility_edge_score") or 70.0),
                "timing_score": float(selected.get("timing_score") or 70.0),
                "risk_reward_score": float(selected.get("risk_reward_score") or 70.0),
                "trade_quality_score": float(selected.get("trade_quality_score") or 90.0),
            }
            try:
                from PIL import Image, ImageDraw
                img = None
                if entry_context and chart_data:
                    width, height = 800, 400
                    # Black background
                    img = Image.new('RGB', (width, height), color=(14, 17, 23))
                    draw = ImageDraw.Draw(img)
                    
                    # Normalize scales
                    x_min = min(c['time'] for c in chart_data)
                    x_max = max(c['time'] for c in chart_data)
                    y_min = min(c['low'] for c in chart_data) * 0.999
                    y_max = max(c['high'] for c in chart_data) * 1.001
                    
                    if x_max == x_min:
                        x_max += 1
                        x_min -= 1
                    if y_max == y_min:
                        y_max += 1
                        y_min -= 1
                    
                    dx = width / (x_max - x_min) if x_max > x_min else 1
                    dy = height / (y_max - y_min) if y_max > y_min else 1
                    
                    # Draw candles
                    for c in chart_data:
                        x = (c['time'] - x_min) * dx
                        o, c_val = (c['open'] - y_min) * dy, (c['close'] - y_min) * dy
                        h, l = (c['high'] - y_min) * dy, (c['low'] - y_min) * dy
                        color = '#26a69a' if c_val >= o else '#ef5350'
                        # wick
                        draw.line([(x, height - h), (x, height - l)], fill=color, width=1)
                        # body
                        draw.rectangle([x - 2, height - max(o, c_val), x + 2, height - min(o, c_val)], fill=color)
                        
                    # Draw EMA5
                    if 'ema5_data' in locals() and ema5_data:
                        for i in range(1, len(ema5_data)):
                            x1, y1 = (ema5_data[i-1]['time'] - x_min) * dx, (ema5_data[i-1]['value'] - y_min) * dy
                            x2, y2 = (ema5_data[i]['time'] - x_min) * dx, (ema5_data[i]['value'] - y_min) * dy
                            draw.line([(x1, height - y1), (x2, height - y2)], fill='#2962FF', width=2)
                            
                    # Draw markers for entry & exit
                    for ts, col, lbl in [(locals().get('entry_ts'), '#26a69a', 'ENTRY'), (locals().get('exit_ts'), '#ef5350', 'EXIT')]:
                        if ts:
                            x = (ts - x_min) * dx
                            draw.line([(x, 0), (x, height)], fill=col, width=1)
                            draw.text((x + 5, 20), lbl, fill=col)
                            
            except Exception as chart_e:
                st.warning(f"Could not generate visual chart screenshot: {chart_e}")
                img = None
                
            res = adapter.get_critique(template, quant_dict, model="", image=img)
            st.markdown(res)
        except Exception as e:
            st.error(f"AI Connection Failed: {e}")
            import traceback
            st.error(traceback.format_exc())

def render_reports():
    st.header("Weekly Report Generation")
    
    if st.button("Generate Weekly Auto Report"):
        provider = st.session_state.get('ai_provider', 'gemini')
        try:
            adapter = AIAdapter(provider=provider)
            with open("weekly_report.txt", "r") as f:
                template = f.read()
                
            df = get_all_trades_df()
            if df.empty:
                st.warning("No trades available to generate a report.")
                return
                
            # Filter to last 7 days of trades if desired, but user asked for "all the data from each trade".
            # To be safe and comprehensive, let's provide the active entries.
            # Clean dataframe for JSON serialization
            df_cleaned = df.copy()
            # Explicit PST formatting to prevent AI temporal confusion
            time_cols = ['entry_time', 'exit_time', 'expiry', 'created_at']
            for col in time_cols:
                if col in df_cleaned.columns:
                    def safe_format(x):
                        if pd.isnull(x): return None
                        try:
                            dt = pd.to_datetime(x)
                            if dt.tzinfo is None:
                                dt = dt.tz_localize('UTC')
                            return dt.tz_convert('America/Los_Angeles').strftime('%Y-%m-%d %H:%M:%S PST')
                        except:
                            return str(x)
                    df_cleaned[col] = df_cleaned[col].apply(safe_format)
            
            # Catch any other remaining datetimes natively
            for col in df_cleaned.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                df_cleaned[col] = df_cleaned[col].astype(str)
                
            df_cleaned = df_cleaned.replace({pd.NA: None, np.nan: None})
            
            df_metrics = df.copy()
            if not df_metrics.empty and "entry_time" in df_metrics.columns:
                df_metrics["entry_time"] = pd.to_datetime(df_metrics["entry_time"])
                if df_metrics["entry_time"].dt.tz is None:
                    df_metrics["entry_time"] = df_metrics["entry_time"].dt.tz_localize("UTC")
                df_metrics["entry_time_pst"] = df_metrics["entry_time"].dt.tz_convert(
                    "America/Los_Angeles"
                )
                df_metrics["trade_date"] = df_metrics["entry_time_pst"].dt.date
                daily_summary = []
                for trade_date, grp in df_metrics.groupby("trade_date"):
                    grp_nonnull = grp.dropna(subset=["pnl"])
                    if grp_nonnull.empty:
                        continue
                    wins = grp_nonnull[grp_nonnull["pnl"] > 0]
                    win_rate = float(len(wins) / len(grp_nonnull)) if len(grp_nonnull) > 0 else 0.0
                    avg_pnl = float(grp_nonnull["pnl"].mean())
                    avg_quality = float(
                        grp_nonnull.get("trade_quality_score", pd.Series([np.nan] * len(grp_nonnull))).mean()
                    )
                    best_idx = grp_nonnull["pnl"].idxmax()
                    worst_idx = grp_nonnull["pnl"].idxmin()
                    best_row = grp_nonnull.loc[best_idx]
                    worst_row = grp_nonnull.loc[worst_idx]
                    def _summarize(row):
                        return {
                            "ticker": str(row.get("ticker", "")),
                            "option_type": str(row.get("option_type", "")),
                            "strike": float(row.get("strike") or 0.0),
                            "pnl": float(row.get("pnl") or 0.0),
                            "trade_quality_score": float(row.get("trade_quality_score") or 0.0),
                            "vol_skew_25d": float(row.get("vol_skew") or 0.0),
                            "r_multiple": float(row.get("r_multiple") or 0.0),
                            "recommended_contracts": int(row.get("recommended_contracts") or 0),
                            "entry_time": str(row.get("entry_time_pst")),
                            "exit_time": str(row.get("exit_time")),
                        }
                    daily_summary.append(
                        {
                            "date": str(trade_date),
                            "win_rate": win_rate,
                            "avg_pnl": avg_pnl,
                            "avg_quality_score": avg_quality,
                            "best_trade": _summarize(best_row),
                            "worst_trade": _summarize(worst_row),
                        }
                    )
            else:
                daily_summary = []
            trades_data = {"trades": daily_summary}
            
            with st.spinner("Analyzing data..."):
                res = adapter.get_critique(template, trades_data)
            st.success("Report Generated:")
            st.markdown(res)
        except Exception as e:
            st.error(f"AI Connection Failed: {e}")

def render_settings():
    st.header("Settings & Secure Keys")
    
    st.subheader("Preferences")
    st.selectbox("Default Market Data Provider", ["yfinance", "polygon (optional)"], key="md_provider")
    st.selectbox("AI Critique Provider", ["noop", "gemini", "groq"], index=1, key="ai_provider")
    st.number_input("Monte Carlo Sample Size", 100, 50000, 10000, key="mc_sims")
    st.number_input("Base Capital for sizing", 50, 100000, 3000, key="capital")
    use_sb = st.checkbox("Use Supabase Hosted DB", value=False, key="use_supabase")
    if use_sb:
        st.caption("Set USE_SUPABASE=true and SUPABASE_URL in environment or secrets and restart the app for this to take effect.")
    
    st.subheader("AI API Keys Secure Store")
    st.info("Keys are AES encryptly stored into the SQLite 'secrets' table using PBKDF2 Master Password or keys from secrets.toml")
    
    master = get_master_key()
    if master:
        st.success("Master Key is Active.")
    else:
        st.error("No Master Key generated! Add MASTER_PASSWORD to secrets.toml or environment variables.")
        st.code("MASTER_PASSWORD = \"your-secure-password\"")
        
    with st.form("api_key_form"):
        p = st.selectbox("Provider", ["gemini_api_key", "groq_api_key", "polygon_api_key"])
        val = st.text_input("Plain Text Key", type="password")
        sub = st.form_submit_button("Encrypt & Store")
        if sub and val:
            if not master:
                st.error("Enable master key first to encrypt.")
            else:
                store_api_key(p, val)
                st.success(f"Stored encrypted key for {p}")
                
    if st.button("Export Encrypted Snippet to GUI"):
        # For user to copy to secrets.toml in cloud if they want
        if not master:
            st.error("No master key")
        else:
            try:
                session = get_session()
                records = session.query(Secret).all()
                snippet = "[ai]\n"
                for r in records:
                    b64_enc = base64.b64encode(r.encrypted_key).decode()
                    snippet += f"{r.provider} = \"{b64_enc}\"\n"
                st.code(snippet, language="toml")
            except Exception as e:
                st.error(f"Error {e}")
