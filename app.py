import os
import streamlit as st
from db import create_tables
import ui as ui

st.set_page_config(
    page_title="SPX 0DTE Journal",
    page_icon="📈",
    layout="wide",
)

# Initialize DB
def init_db():
    try:
        create_tables()
    except Exception as e:
        print(f"Database init error: {e}")

init_db()

def apply_terminal_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap');
        
        /* Core App styling */
        .stApp {
            background-color: #0a0a0a !important;
        }
        
        * {
            font-family: 'Fira Code', Courier, monospace !important;
        }
        
        /* Text Colors */
        h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stText {
            color: #d3d3d3 !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.8rem !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #a0a0a0 !important;
        }
        
        /* Inputs and Selectboxes */
        div.stTextInput > div > div > input, 
        div.stNumberInput > div > div > input,
        .stSelectbox > div > div > div, 
        textarea {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            border-radius: 0px !important;
            font-size: 0.85rem !important;
        }
        
        /* Selectbox dropdown items specifically */
        ul[role="listbox"] li {
            font-size: 0.85rem !important;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #050505 !important;
            border-right: 2px solid #333333 !important;
        }
        
        /* Buttons */
        button[kind="primary"] {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            border-radius: 0 !important;
            text-transform: uppercase !important;
            font-weight: bold !important;
        }
        button[kind="primary"]:hover {
            background-color: #555555 !important;
            color: #ffffff !important;
            border-color: #777777 !important;
        }
        button[kind="secondary"] {
            background-color: #000000 !important;
            color: #d3d3d3 !important;
            border: 1px dashed #555555 !important;
            border-radius: 0 !important;
        }
        button[kind="secondary"]:hover {
            background-color: #333333 !important;
            color: #ffffff !important;
            border-style: solid !important;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: 1px solid #555555;
        }
        
        /* Success / Exception messages */
        div[data-testid="stAlert"] {
            background-color: #1a1a1a !important;
            color: #d3d3d3 !important;
            border-left: 4px solid #555555 !important;
        }
        div[data-testid="stAlert"]:has(div.st-emotion-cache-1kqj0k3) {
            /* Error */
            background-color: #3b0e0e !important;
            color: #ff3333 !important;
            border-left: 4px solid #ff3333 !important;
        }
        div[data-testid="stAlert"]:has(div.st-emotion-cache-12t9kfb) {
            /* Warning / info */
            background-color: #332b00 !important;
            color: #ffcc00 !important;
            border-left: 4px solid #ffcc00 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    apply_terminal_theme()
    st.sidebar.title("SPX 0DTE Journal")
    
    # Power User Command Line Input
    cmd = st.sidebar.text_input("Terminal Input (e.g. 'go new trade')", key="cli_input").lower().strip()
    
    pages = {
        "dashboard": ("Dashboard", ui.render_dashboard),
        "new trade": ("New Trade", ui.render_new_trade),
        "viewer": ("Trade Viewer", ui.render_trade_viewer),
        "ai": ("AI Reports", ui.render_reports),
        "settings": ("Settings & Keys", ui.render_settings)
    }
    
    # Process CLI command
    if cmd.startswith("go "):
        target = cmd.replace("go ", "").strip()
        for k, (display, func) in pages.items():
            if k in target or target in k:
                st.session_state.nav_selection = display
                break
    
    page_names = [v[0] for v in pages.values()]
    
    selection = st.sidebar.radio(
        "Navigation", 
        page_names, 
        index=page_names.index(st.session_state.get('nav_selection', 'Dashboard')) if 'nav_selection' in st.session_state else 0,
        key="nav_radio"
    )
    
    # Remember selection
    st.session_state.nav_selection = selection
    
    # Find matching func
    for k, (display, func) in pages.items():
        if display == selection:
            func()
            break

if __name__ == "__main__":
    main()
