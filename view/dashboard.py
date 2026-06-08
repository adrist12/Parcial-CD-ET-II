import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============= CONFIG STREAMLIT =============
st.set_page_config(
    page_title="Análisis Predictivo de personas en situacion de calle",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)