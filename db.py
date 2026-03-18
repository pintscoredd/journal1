import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Text, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

_ENGINE = None

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_uuid = Column(String, unique=True)
    ticker = Column(String)
    option_type = Column(String)  # 'call' or 'put'
    strike = Column(Float)
    expiry = Column(Date)
    contracts = Column(Integer)
    
    entry_price = Column(Float)
    exit_price = Column(Float)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    
    entry_note = Column(Text)
    exit_note = Column(Text)
    tags = Column(String)
    
    underlying_entry_price = Column(Float)
    underlying_exit_price = Column(Float)
    
    vix_at_entry = Column(Float)
    realized_vol_5m = Column(Float)
    realized_vol_15m = Column(Float)
    
    implied_vol_entry = Column(Float)
    implied_vol_exit = Column(Float)
    
    # New Volatility Surface Proxies
    vol_skew = Column(Float)
    vol_term_slope = Column(Float)
    
    delta_entry = Column(Float)
    gamma_entry = Column(Float)
    theta_entry = Column(Float)
    vega_entry = Column(Float)
    
    delta_exit = Column(Float)
    gamma_exit = Column(Float)
    theta_exit = Column(Float)
    vega_exit = Column(Float)
    
    entry_theoretical_price = Column(Float)
    exit_theoretical_price = Column(Float)
    
    pnl = Column(Float)
    max_loss = Column(Float)
    r_multiple = Column(Float)
    hold_time_minutes = Column(Float)
    vol_ratio = Column(Float)
    
    # Component Scores
    entry_execution_score = Column(Float)
    volatility_edge_score = Column(Float)
    timing_score = Column(Float)
    risk_reward_score = Column(Float)
    
    # Final numeric score
    trade_quality_score = Column(Float)
    
    # Kelly Sizing Suggestion
    recommended_contracts = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AICache(Base):
    __tablename__ = 'ai_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String, unique=True)  # Can be sha256(prompt + quant_json + model)
    provider = Column(String)
    prompt_hash = Column(String) # sha256 of the exact prompt block
    response_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Secret(Base):
    __tablename__ = 'secrets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, unique=True)
    encrypted_key = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

# Engine initialization and config
def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    use_supabase = os.environ.get("USE_SUPABASE", "false").lower() == "true"
    if use_supabase:
        database_url = os.environ.get("SUPABASE_URL", "")
        if not database_url:
            raise ValueError("SUPABASE_URL must be provided when USE_SUPABASE is true")
    else:
        data_dir = os.path.expanduser('~/.spx_0dte_journal')
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, 'journal.db')
        
        old_local_db = os.path.join(os.path.dirname(__file__), 'journal.db')
        if not os.path.exists(db_path) and os.path.exists(old_local_db):
            import shutil
            try:
                shutil.copy2(old_local_db, db_path)
            except Exception:
                pass

        database_url = f"sqlite:///{db_path}"
    
    _ENGINE = create_engine(database_url)
    return _ENGINE

def create_tables(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()
