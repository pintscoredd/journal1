import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import streamlit as st
from db import get_session, Secret

def get_master_key():
    # Fallback cascade: st.secrets > environment
    # Support both raw base64 key or PBKDF2 derived from password
    
    # Check if raw MASTER_KEY is provided
    try:
        if "MASTER_KEY" in st.secrets.get("app", {}):
            return st.secrets["app"]["MASTER_KEY"].encode()
    except Exception:
        pass
        
    master_key = os.environ.get("MASTER_KEY")
    if master_key:
        return master_key.encode()
        
    # Check for MASTER_PASSWORD
    try:
        if "MASTER_PASSWORD" in st.secrets.get("app", {}):
            pwd = st.secrets["app"]["MASTER_PASSWORD"].encode()
            # Derive key securely using a static salt for the DB (in a real app, salt should be stored, 
            # but for a local single-user SQLite DB, this is sufficient for non-plain text).
            salt = b'spx_0dte_journal_salt_2024'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            return base64.urlsafe_b64encode(kdf.derive(pwd))
    except Exception:
        pass
        
    pwd = os.environ.get("MASTER_PASSWORD")
    if pwd:
        salt = b'spx_0dte_journal_salt_2024'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(pwd.encode()))
        
    return None

def encrypt_key(plain_key: str) -> bytes:
    master_key = get_master_key()
    if not master_key:
        raise ValueError("Master key or password not configured")
    
    f = Fernet(master_key)
    return f.encrypt(plain_key.encode())

def decrypt_key(encrypted_key: bytes) -> str:
    master_key = get_master_key()
    if not master_key:
        raise ValueError("Master key or password not configured")
        
    f = Fernet(master_key)
    return f.decrypt(encrypted_key).decode()

def get_api_key(provider: str) -> str:
    # 1. st.secrets
    try:
        # Check root level first
        if provider in st.secrets:
            return st.secrets[provider]
            
        # Then categories
        if provider in st.secrets.get("ai", {}):
            return st.secrets["ai"][provider]
        if provider in st.secrets.get("market_data", {}):
            return st.secrets["market_data"][provider]
    except Exception:
        pass

    # 2. env
    env_var = f"{provider.upper()}"
    if env_var in os.environ:
        return os.environ[env_var]
        
    # 3. DB
    session = get_session()
    try:
        secret_record = session.query(Secret).filter_by(provider=provider).first()
        if secret_record and secret_record.encrypted_key:
            return decrypt_key(secret_record.encrypted_key)
    except Exception:
        pass
    finally:
        session.close()

    return ""

def store_api_key(provider: str, plain_key: str):
    session = get_session()
    try:
        encrypted = encrypt_key(plain_key)
        record = session.query(Secret).filter_by(provider=provider).first()
        if record:
            record.encrypted_key = encrypted
        else:
            record = Secret(provider=provider, encrypted_key=encrypted)
            session.add(record)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
