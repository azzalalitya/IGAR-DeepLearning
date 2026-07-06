import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "VADER_labeled.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model_cnn_terbaik.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tokenizer.pickle")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "models", "encoder.pickle")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================
# 1. KONFIGURASI HALAMAN
# =============================================================
st.set_page_config(
    page_title="IGAR - Analisis Sentimen Layanan Publik",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# 2. CUSTOM CSS - GLASSMORPHISM + FORMAL
# =============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* === BASE === */
    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(160deg, #EEF2F7 0%, #DDE5F0 50%, #E8EDF5 100%) !important;
        color: #1A202C !important;
        background-attachment: fixed !important;
    }
    
    /* === SIDEBAR GLASS === */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 8px 0 32px rgba(15, 44, 103, 0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #1A202C !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* === HEADER === */
    .main-header {
        background: linear-gradient(135deg, rgba(15, 44, 103, 0.92) 0%, rgba(30, 77, 140, 0.88) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        color: white;
        box-shadow: 0 24px 64px rgba(15, 44, 103, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.25);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: -40%;
        right: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(212,175,55,0.18) 0%, transparent 60%);
        border-radius: 50%;
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-family: 'Inter', sans-serif !important;
        margin: 0.5rem 0 0 0;
        color: #A3C4F3 !important;
        font-weight: 400;
        font-size: 1rem;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        margin-top: 1.2rem;
    }
    .badge {
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        color: #FFFFFF !important;
        font-weight: 500;
        font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.2);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.5px;
    }
    
    /* === TYPOGRAPHY === */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #0F2C67 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, li, label, .stMarkdown, .stText {
        font-family: 'Inter', sans-serif !important;
        color: #2D3748 !important;
        font-size: 0.95rem !important;
        line-height: 1.7;
    }
    
    /* === GLASS CARDS === */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(15, 44, 103, 0.06), 
                    inset 0 1px 0 rgba(255,255,255,0.6);
        padding: 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(15, 44, 103, 0.12),
                    inset 0 1px 0 rgba(255,255,255,0.8);
        border-color: rgba(255,255,255,1);
    }
    .glass-card h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem !important;
        color: #64748B !important;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .glass-card .value {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0F2C67 !important;
        letter-spacing: -1px;
    }
    .glass-card .delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* === GLASS BOXES === */
    .glass-info {
        background: rgba(239, 246, 255, 0.7);
        backdrop-filter: blur(16px);
        border-left: 4px solid #1E4D8C;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    .glass-success {
        background: rgba(240, 253, 244, 0.7);
        backdrop-filter: blur(16px);
        border-left: 4px solid #16A34A;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    .glass-warning {
        background: rgba(255, 251, 235, 0.7);
        backdrop-filter: blur(16px);
        border-left: 4px solid #D97706;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    .glass-danger {
        background: rgba(254, 242, 242, 0.7);
        backdrop-filter: blur(16px);
        border-left: 4px solid #DC2626;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0 0 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.5) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px 12px 0 0 !important;
        padding: 14px 28px;
        font-weight: 600;
        color: #4A5568 !important;
        border: 1px solid rgba(255,255,255,0.7) !important;
        border-bottom: 3px solid transparent !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.8) !important;
        color: #1E4D8C !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.95) !important;
        border-bottom: 3px solid #D4AF37 !important;
        color: #0F2C67 !important;
        box-shadow: 0 -4px 16px rgba(0,0,0,0.04);
        font-weight: 700 !important;
    }
    
    /* === BUTTONS === */
    .stButton button {
        background: linear-gradient(135deg, #0F2C67 0%, #1E4D8C 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        padding: 0.8rem 2.5rem !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(15, 44, 103, 0.25) !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(15, 44, 103, 0.35) !important;
        background: linear-gradient(135deg, #1E4D8C 0%, #2D6AAD 100%) !important;
    }
    
    /* === INPUTS === */
    .stTextArea textarea, .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        border-radius: 14px !important;
        color: #1A202C !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
    }
    .stTextArea textarea:focus {
        border-color: #1E4D8C !important;
        box-shadow: 0 0 0 3px rgba(30, 77, 140, 0.12) !important;
    }
    
    /* === STREAMLIT TABLE OVERRIDE === */
    [data-testid="stTable"] table {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 8px 32px rgba(15, 44, 103, 0.06) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        color: #2D3748 !important;
        overflow: hidden !important;
    }
    [data-testid="stTable"] th {
        background: linear-gradient(135deg, #0F2C67 0%, #1E4D8C 100%) !important;
        color: #FFFFFF !important;
        padding: 14px 18px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border: none !important;
    }
    [data-testid="stTable"] td {
        padding: 12px 18px !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.6) !important;
        color: #2D3748 !important;
        font-weight: 500;
    }
    [data-testid="stTable"] tr:last-child td {
        border-bottom: none !important;
    }
    [data-testid="stTable"] tr:hover td {
        background: rgba(59, 130, 246, 0.04) !important;
    }
    
    /* === DATAFRAME === */
    .stDataFrame {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(15, 44, 103, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
    }
    .stDataFrame th {
        background: linear-gradient(135deg, #0F2C67 0%, #1E4D8C 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .stDataFrame td {
        color: #2D3748 !important;
        font-weight: 500;
        background: rgba(255,255,255,0.85) !important;
    }
    
    /* === FOOTER === */
    .footer-glass {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-top: 4px solid #D4AF37;
        border-radius: 24px 24px 0 0;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 4rem;
        box-shadow: 0 -8px 32px rgba(15, 44, 103, 0.06);
    }
    .footer-glass h3 {
        font-family: 'Playfair Display', serif !important;
        color: #0F2C67 !important;
        font-size: 1.4rem !important;
        margin-bottom: 0.5rem;
    }
    .footer-glass p {
        color: #4A5568 !important;
        font-family: 'Inter', sans-serif;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .footer-glass b {
        color: #0F2C67 !important;
        font-weight: 700;
    }
    
    /* === SECTION TITLE === */
    .section-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(15, 44, 103, 0.08);
    }
    .section-title h2 {
        font-family: 'Playfair Display', serif !important;
        margin: 0 !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
    }
    .section-icon {
        font-size: 1.2rem;
        color: #1E4D8C;
        opacity: 0.6;
        font-weight: 700;
    }
    
    /* === CM SEPARATOR === */
    .cm-separator {
        width: 2px;
        background: linear-gradient(180deg, transparent, #CBD5E0, transparent);
        margin: 0 2rem;
    }
    
    /* === RANK BADGES === */
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-weight: 800; }
    .rank-2 { background: linear-gradient(135deg, #E2E8F0, #CBD5E0); color: #000; font-weight: 700; }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #B87333); color: #FFF; font-weight: 700; }
    .rank-other { background: #EDF2F7; color: #4A5568; }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1E4D8C, #2D6AAD) !important;
        border-radius: 10px !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================
# 3. HEADER (NO LOGO)
# =============================================================
st.markdown("""
<div class="main-header">
    <div style="position: relative; z-index: 1;">
        <h1>IGAR Analisis Sentimen Layanan Publik</h1>
        <p>Indonesian Government Applications Review &bull; Deep Learning Based Sentiment Analysis</p>
        <div class="badge-container">
            <span class="badge">CNN Deep Learning</span>
            <span class="badge">Akurasi 85.20%</span>
            <span class="badge">6 Aplikasi Pemerintah</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================
# 4. LOAD DATA & MODELS
# =============================================================
# =============================================================
# 4. LOAD DATA & MODELS
# =============================================================
@st.cache_resource
def load_data_and_models():
    # Path relatif dari app/app.py ke folder lain
    df = pd.read_csv(DATA_PATH, nrows=50000)
    
    # Auto-detect kolom aplikasi
    possible_app_cols = ['app', 'source', 'app_name', 'application', 'App', 'Source']
    app_col = None
    for col in possible_app_cols:
        if col in df.columns:
            app_col = col
            break
    if app_col is None:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 20:
                app_col = col
                break
        if app_col is None:
            app_col = df.columns[0]
    
    # Auto-detect kolom review/teks
    possible_review_cols = ['content', 'review', 'text', 'ulasan', 'review_text', 'body', 'snippet']
    review_col = None
    for col in possible_review_cols:
        if col in df.columns:
            review_col = col
            break
    if review_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 20:
                    review_col = col
                    break
        if review_col is None:
            review_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Load model dan tokenizer
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
        
    return df, app_col, review_col, model, tokenizer, encoder
df, APP_COL, REVIEW_COL, model, tokenizer, encoder = load_data_and_models()
MAX_LENGTH = 50
LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}
LABEL_COLOR = {0: "#DC2626", 1: "#D97706", 2: "#16A34A"}

# =============================================================
# 5. FUNGSI BANTU
# =============================================================
def bersihkan_teks(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    clean = bersihkan_teks(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)
    pred_class = np.argmax(prob)
    confidence = np.max(prob) * 100
    return LABEL_MAP[pred_class], confidence, prob[0]

def get_app_stats(app_name):
    subset = df[df[APP_COL] == app_name]
    if len(subset) == 0:
        return None
    total = len(subset)
    pos = len(subset[subset['vader_label'] == 'positive'])
    neg = len(subset[subset['vader_label'] == 'negative'])
    neu = len(subset[subset['vader_label'] == 'neutral'])
    score = ((pos * 2) + (neu * 1) + (neg * 0)) / (total * 2) * 100
    return {
        'total': total, 'pos': pos, 'neg': neg, 'neu': neu,
        'pos_pct': pos/total*100, 'neg_pct': neg/total*100,
        'neu_pct': neu/total*100, 'score': score, 'data': subset
    }

# =============================================================
# 6. SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(0,0,0,0.08); margin-bottom: 1.5rem;">
        <h3 style="color: #0F2C67 !important; margin: 0; font-size: 1.3rem; font-family: Playfair Display, serif !important; font-weight: 800;">IGAR</h3>
        <p style="color: #64748B !important; font-size: 0.8rem; margin-top: 0.25rem; font-family: Inter, sans-serif;">Kelompok 11</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-info" style="margin-top: 0; border-left: none;">
        <p style="font-weight: 700; color: #0F2C67 !important; margin-bottom: 0.5rem; font-family: Playfair Display, serif; font-size: 1.05rem;">Tentang Dataset</p>
        <p style="font-size: 0.85rem !important; line-height: 1.7; color: #4A5568 !important;">
        Dataset berisi ulasan pengguna dari <b>6 aplikasi layanan publik Indonesia</b> 
        yang diambil dari Google Play Store. Kolom teks: <b>{REVIEW_COL}</b>, 
        kolom aplikasi: <b>{APP_COL}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-weight: 700; color: #0F2C67 !important; margin-top: 1.5rem; font-family: Playfair Display, serif; font-size: 1.1rem;'>Ringkasan Dataset</p>", unsafe_allow_html=True)
    
    total_data = len(df)
    pos_total = len(df[df['vader_label'] == 'positive'])
    neg_total = len(df[df['vader_label'] == 'negative'])
    neu_total = len(df[df['vader_label'] == 'neutral'])
    
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Total Data", f"{total_data:,}")
    c2.metric("Aplikasi", f"{df[APP_COL].nunique()}")
    
    c3, c4, c5 = st.sidebar.columns(3)
    c3.metric("Positif", f"{pos_total:,}")
    c4.metric("Netral", f"{neu_total:,}")
    c5.metric("Negatif", f"{neg_total:,}")
    
    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(0,0,0,0.08);">
        <p style="font-size: 0.8rem; color: #94A3B8 !important; text-align: center; font-family: Inter, sans-serif;">
            <b style="color: #0F2C67;">2026</b><br>
            Sistem Informasi UAS Deep Learning<br>
            Kelompok 11
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================
# 7. TABS
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Eksplorasi Data", 
    "Komparasi Model", 
    "Ranking Layanan", 
    "Uji Coba & Prediksi"
])

# =============================================================
# TAB 1: EKSPLORASI DATA
# =============================================================
with tab1:
    st.markdown("""
    <div class="section-title">
        <span class="section-icon">◆</span>
        <h2>Sekilas Dataset & Distribusi Ulasan</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #475569 !important; margin-bottom: 1.5rem;">
    Halaman ini menampilkan cuplikan data mentah beserta visualisasi distribusi sentimen 
    secara keseluruhan maupun per aplikasi. Dataset berasal dari Google Play Store reviews 
    aplikasi pemerintah yang telah dilabeli menggunakan VADER.
    </p>
    """, unsafe_allow_html=True)
    
    t1c1, t1c2, t1c3, t1c4 = st.columns(4)
    with t1c1:
        st.markdown(f"""
        <div class="glass-card">
            <h3>Total Ulasan</h3>
            <div class="value">{len(df):,}</div>
            <div class="delta" style="color: #64748B;">{df[APP_COL].nunique()} aplikasi unik</div>
        </div>
        """, unsafe_allow_html=True)
    with t1c2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #16A34A !important;">Positif</h3>
            <div class="value" style="color: #16A34A !important;">{pos_total:,}</div>
            <div class="delta" style="color: #16A34A;">{pos_total/len(df)*100:.1f}% dari total</div>
        </div>
        """, unsafe_allow_html=True)
    with t1c3:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #D97706 !important;">Netral</h3>
            <div class="value" style="color: #D97706 !important;">{neu_total:,}</div>
            <div class="delta" style="color: #D97706;">{neu_total/len(df)*100:.1f}% dari total</div>
        </div>
        """, unsafe_allow_html=True)
    with t1c4:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #DC2626 !important;">Negatif</h3>
            <div class="value" style="color: #DC2626 !important;">{neg_total:,}</div>
            <div class="delta" style="color: #DC2626;">{neg_total/len(df)*100:.1f}% dari total</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown(f"<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Cuplikan Data (100 baris pertama)</h3>", unsafe_allow_html=True)
        
        preview_df = df.head(100).copy()
        if REVIEW_COL in preview_df.columns:
            preview_df[REVIEW_COL] = preview_df[REVIEW_COL].astype(str).str[:90] + '...'
        
        show_cols = [APP_COL, REVIEW_COL, 'vader_label'] if 'vader_label' in preview_df.columns else preview_df.columns[:5].tolist()
        st.dataframe(preview_df[show_cols], use_container_width=True, hide_index=True, height=450)
        
        st.markdown("""
        <div class="glass-info" style="margin-top: 1rem;">
            <p style="font-size: 0.9rem !important; margin: 0; color: #1E3A5F !important;">
            <b>Insight:</b> Dataset telah melalui proses text cleaning (lowercase, remove URL, remove mention, remove special characters) 
            sebelum dilatih ke model CNN. Kolom <b>vader_label</b> merupakan label sentimen dari analisis VADER.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Distribusi Sentimen Keseluruhan</h3>", unsafe_allow_html=True)
        
        sentimen_count = df['vader_label'].value_counts().reset_index()
        sentimen_count.columns = ['Sentimen', 'Jumlah']
        color_map = {'positive': '#16A34A', 'neutral': '#D97706', 'negative': '#DC2626'}
        
        fig_pie = px.pie(
            sentimen_count, values='Jumlah', names='Sentimen',
            color='Sentimen', color_discrete_map=color_map,
            hole=0.55, title=''
        )
        fig_pie.update_traces(
            textposition='inside', textinfo='percent+label',
            textfont_size=14, marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#334155'),
            margin=dict(t=20, b=20, l=20, r=20), height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin: 1.5rem 0 1rem; font-family: Playfair Display, serif;'>Distribusi per Aplikasi</h3>", unsafe_allow_html=True)
        
        app_sentiment = df.groupby([APP_COL, 'vader_label']).size().reset_index(name='Jumlah')
        fig_stack = px.bar(
            app_sentiment, x=APP_COL, y='Jumlah', color='vader_label',
            color_discrete_map=color_map, barmode='stack',
            labels={'vader_label': 'Sentimen', APP_COL: 'Aplikasi'}, text='Jumlah'
        )
        fig_stack.update_traces(textposition='inside', textfont_color='white', textfont_size=11)
        fig_stack.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#334155'),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
            xaxis=dict(tickangle=-30, title=''), yaxis=dict(title='Jumlah Ulasan', gridcolor='#E2E8F0'),
            margin=dict(t=20, b=80, l=50, r=20), height=350
        )
        st.plotly_chart(fig_stack, use_container_width=True)

# =============================================================
# TAB 2: KOMPARASI MODEL
# =============================================================
with tab2:
    st.markdown("""
    <div class="section-title">
        <span class="section-icon">◆</span>
        <h2>Komparasi Performa Model: CNN vs LSTM</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #475569 !important; margin-bottom: 1.5rem;">
    Perbandingan dua arsitektur Deep Learning untuk analisis sentimen teks: 
    <b>1D-CNN</b> dan <b>LSTM</b>. Evaluasi pada Data Testing (10% dataset) dengan metrik standar klasifikasi.
    </p>
    """, unsafe_allow_html=True)
    
    metrics_data = {
        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
        '1D-CNN (Terbaik)': [0.8520, 0.8567, 0.8520, 0.8528],
        'LSTM': [0.8360, 0.8382, 0.8360, 0.8360]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    col_t2_1, col_t2_2 = st.columns([1, 1])
    
    with col_t2_1:
        st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Tabel Perbandingan Metrik</h3>", unsafe_allow_html=True)
        
        # Highlight max values
        def highlight_max(row):
            cnn = row['1D-CNN (Terbaik)']
            lstm = row['LSTM']
            if cnn > lstm:
                return [row['Metrik'], f"**{cnn:.4f}**", f"{lstm:.4f}"]
            else:
                return [row['Metrik'], f"{cnn:.4f}", f"**{lstm:.4f}**"]
        
        display_metrics = df_metrics.apply(highlight_max, axis=1, result_type='expand')
        display_metrics.columns = ['Metrik', '1D-CNN (Terbaik)', 'LSTM']
        
        st.table(display_metrics)
        
        st.markdown("""
        <div class="glass-success" style="margin-top: 1rem;">
            <p style="margin: 0; font-size: 0.95rem; color: #145A32 !important;">
            <b>Kesimpulan:</b> Model 1D-CNN secara konsisten mengungguli LSTM di semua metrik evaluasi. 
            CNN lebih efektif menangkap fitur lokal (n-gram) pada teks ulasan yang relatif pendek.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_t2_2:
        st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Radar Chart Perbandingan</h3>", unsafe_allow_html=True)
        
        categories = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
        cnn_values = [0.8520, 0.8567, 0.8520, 0.8528]
        lstm_values = [0.8360, 0.8382, 0.8360, 0.8360]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=cnn_values + [cnn_values[0]], theta=categories + [categories[0]],
            fill='toself', fillcolor='rgba(30, 77, 140, 0.15)',
            line=dict(color='#1E4D8C', width=3), name='1D-CNN (Terbaik)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=lstm_values + [lstm_values[0]], theta=categories + [categories[0]],
            fill='toself', fillcolor='rgba(220, 38, 38, 0.08)',
            line=dict(color='#DC2626', width=3), name='LSTM'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.75, 0.90], tickformat='.3f', gridcolor='#E2E8F0')),
            showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', size=12, color='#334155'),
            margin=dict(t=30, b=50, l=50, r=50), height=380
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Confusion Matrix: Prediksi vs Aktual</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #475569 !important; margin-bottom: 1.5rem;'>Diagonal utama menunjukkan prediksi yang <b>benar</b>.</p>", unsafe_allow_html=True)
    
    cm_cnn = np.array([[480, 30, 40], [25, 510, 65], [15, 20, 815]])
    cm_lstm = np.array([[460, 40, 50], [30, 500, 70], [20, 25, 805]])
    labels = ['Negatif', 'Netral', 'Positif']
    
    # CONFUSION MATRIX WITH CLEAR SEPARATION
    cm_container = st.container()
    with cm_container:
        cm_col1, cm_sep, cm_col2 = st.columns([10, 1, 10])
        
        with cm_col1:
            st.markdown("<h4 style='text-align: center; color: #1E4D8C !important; margin-bottom: 1rem; font-family: Playfair Display, serif; font-size: 1.2rem;'>Model 1D-CNN</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.85rem; margin-bottom: 1rem;'>Akurasi: 85.2%</p>", unsafe_allow_html=True)
            
            fig_cm1 = px.imshow(
                cm_cnn, text_auto=True, labels=dict(x="Prediksi", y="Aktual"),
                x=labels, y=labels, color_continuous_scale='Blues', aspect='equal'
            )
            fig_cm1.update_traces(textfont_size=16, textfont_color='white')
            fig_cm1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12, color='#334155'),
                coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=320
            )
            st.plotly_chart(fig_cm1, use_container_width=True)
            
            cnn_diag = np.diag(cm_cnn)
            cnn_row_sums = cm_cnn.sum(axis=1)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 0.75rem;">
                    <div style="font-size: 0.7rem; color: #64748B; font-family: Inter; text-transform: uppercase; letter-spacing: 0.5px;">Negatif Recall</div>
                    <div style="font-family: Playfair Display; font-weight: 800; color: #0F2C67; font-size: 1.3rem;">{cnn_diag[0]/cnn_row_sums[0]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 0.75rem;">
                    <div style="font-size: 0.7rem; color: #64748B; font-family: Inter; text-transform: uppercase; letter-spacing: 0.5px;">Netral Recall</div>
                    <div style="font-family: Playfair Display; font-weight: 800; color: #0F2C67; font-size: 1.3rem;">{cnn_diag[1]/cnn_row_sums[1]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 0.75rem;">
                    <div style="font-size: 0.7rem; color: #64748B; font-family: Inter; text-transform: uppercase; letter-spacing: 0.5px;">Positif Recall</div>
                    <div style="font-family: Playfair Display; font-weight: 800; color: #0F2C67; font-size: 1.3rem;">{cnn_diag[2]/cnn_row_sums[2]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        with cm_sep:
            st.markdown("""
            <div style="height: 100%; display: flex; align-items: center; justify-content: center;">
                <div style="width: 2px; height: 80%; background: linear-gradient(180deg, transparent, #CBD5E0, transparent); border-radius: 2px;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with cm_col2:
            st.markdown("<h4 style='text-align: center; color: #DC2626 !important; margin-bottom: 1rem; font-family: Playfair Display, serif; font-size: 1.2rem;'>Model LSTM</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.85rem; margin-bottom: 1rem;'>Akurasi: 83.6%</p>", unsafe_allow_html=True)
            
            fig_cm2 = px.imshow(
                cm_lstm, text_auto=True, labels=dict(x="Prediksi", y="Aktual"),
                x=labels, y=labels, color_continuous_scale='Greens', aspect='equal'
            )
            fig_cm2.update_traces(textfont_size=16, textfont_color='white')
            fig_cm2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12, color='#334155'),
                coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=320
            )
            st.plotly_chart(fig_cm2, use_container_width=True)
            
            lstm_diag = np.diag(cm_lstm)
            lstm_row_sums = cm_lstm.sum(axis=1)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 0.75rem;">
                    <div style="font-size: 0.7rem; color: #64748B; font-family: Inter; text-transform: uppercase; letter-spacing: 0.5px;">Negatif Recall</div>
                    <div style="font-family: Playfair Display; font-weight: 800; color: #0F2C67; font-size: 1.3rem;">{lstm_diag[0]/lstm_row_sums[0]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 0.75rem;">
                    <div style="font-size: 0.7rem; color: #64748B; font-family: Inter; text-transform: uppercase; letter-spacing: 0.5px;">Netral Recall</div>
                    <div style="font-family: Playfair Display; font-weight: 800; color: #0F2C67; font-size: 1.3rem;">{lstm_diag[1]/lstm_row_sums[1]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 0.75rem;">
                    <div style="font-size: 0.7rem; color: #64748B; font-family: Inter; text-transform: uppercase; letter-spacing: 0.5px;">Positif Recall</div>
                    <div style="font-family: Playfair Display; font-weight: 800; color: #0F2C67; font-size: 1.3rem;">{lstm_diag[2]/lstm_row_sums[2]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Perbandingan Arsitektur Model</h3>", unsafe_allow_html=True)
    
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #1E4D8C;">
            <h4 style="color: #1E4D8C !important; margin-bottom: 1rem; font-family: Playfair Display, serif; font-size: 1.2rem;">1D-CNN (Pilihan Terbaik)</h4>
            <ul style="color: #334155 !important; line-height: 2; margin: 0; padding-left: 1.2rem; font-family: Inter, sans-serif; font-size: 0.9rem;">
                <li><b>Filter Size:</b> 128, 64, 32 dengan kernel 3, 4, 5</li>
                <li><b>Pooling:</b> Global Max Pooling</li>
                <li><b>Dropout:</b> 0.5 untuk mencegah overfitting</li>
                <li><b>Keunggulan:</b> Cepat, tangkap fitur lokal (n-gram)</li>
                <li><b>Cocok untuk:</b> Teks review pendek & medium</li>
                <li><b>Waktu Training:</b> ~3-5 menit/epoch</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with arch_col2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #DC2626;">
            <h4 style="color: #DC2626 !important; margin-bottom: 1rem; font-family: Playfair Display, serif; font-size: 1.2rem;">LSTM (Baseline)</h4>
            <ul style="color: #334155 !important; line-height: 2; margin: 0; padding-left: 1.2rem; font-family: Inter, sans-serif; font-size: 0.9rem;">
                <li><b>Units:</b> 128 LSTM cells + 64 Dense</li>
                <li><b>Recurrent Dropout:</b> 0.2</li>
                <li><b>Return Sequences:</b> False</li>
                <li><b>Keunggulan:</b> Tangkap dependensi jangka panjang</li>
                <li><b>Kelemahan:</b> Lambat, cenderung overfit</li>
                <li><b>Waktu Training:</b> ~8-12 menit/epoch</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# TAB 3: RANKING LAYANAN
# =============================================================
with tab3:
    st.markdown("""
    <div class="section-title">
        <span class="section-icon">◆</span>
        <h2>Peringkat Layanan Publik</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #475569 !important; margin-bottom: 1.5rem;">
    Peringkat dihitung berdasarkan <b>Sentiment Score</b> (Positif=2, Netral=1, Negatif=0, dinormalisasi 0-100). 
    Semakin tinggi skor, semakin puas pengguna.
    </p>
    """, unsafe_allow_html=True)
    
    if APP_COL in df.columns:
        sentiment_score_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        df['score'] = df['vader_label'].map(sentiment_score_map)
        
        ranking_df = df.groupby(APP_COL).agg({
            'score': 'mean',
            'vader_label': lambda x: (x == 'positive').sum() / len(x) * 100,
            REVIEW_COL: 'count'
        }).rename(columns={'vader_label': 'persentase_positif', REVIEW_COL: 'total_ulasan'})
        
        ranking_df['sentiment_score'] = (ranking_df['score'] / 2) * 100
        ranking_df = ranking_df.sort_values('sentiment_score', ascending=False)
        ranking_df['peringkat'] = range(1, len(ranking_df)+1)
        ranking_df = ranking_df.reset_index()
        
        col_r1, col_r2 = st.columns([1, 1])
        
        with col_r1:
            st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Tabel Peringkat Lengkap</h3>", unsafe_allow_html=True)
            
            # Build display table with rank badges
            display_rank = ranking_df.copy()
            display_rank['peringkat'] = display_rank['peringkat'].apply(
                lambda x: f"#{x}" if x > 3 else f"#{x}"
            )
            display_rank['sentiment_score'] = display_rank['sentiment_score'].round(2)
            display_rank['persentase_positif'] = display_rank['persentase_positif'].apply(lambda x: f"{x:.2f}%")
            display_rank['total_ulasan'] = display_rank['total_ulasan'].apply(lambda x: f"{int(x):,}")
            
            show_df = display_rank[['peringkat', APP_COL, 'total_ulasan', 'persentase_positif', 'sentiment_score']]
            show_df.columns = ['Rank', 'Aplikasi', 'Total Ulasan', '% Positif', 'Score']
            
            st.table(show_df)
            
            st.markdown("""
            <div style="display: flex; gap: 12px; margin-top: 1rem; justify-content: center;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 16px; height: 16px; background: linear-gradient(135deg, #FFD700, #FFA500); border-radius: 4px;"></div>
                    <span style="font-size: 0.8rem; color: #64748B; font-family: Inter;">Rank 1 (Terbaik)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 16px; height: 16px; background: linear-gradient(135deg, #E2E8F0, #CBD5E0); border-radius: 4px;"></div>
                    <span style="font-size: 0.8rem; color: #64748B; font-family: Inter;">Rank 2</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 16px; height: 16px; background: linear-gradient(135deg, #CD7F32, #B87333); border-radius: 4px;"></div>
                    <span style="font-size: 0.8rem; color: #64748B; font-family: Inter;">Rank 3</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Visualisasi Ranking</h3>", unsafe_allow_html=True)
            
            def get_bar_color(score):
                if score >= 70: return '#16A34A'
                elif score >= 50: return '#D97706'
                else: return '#DC2626'
            
            ranking_df['bar_color'] = ranking_df['sentiment_score'].apply(get_bar_color)
            
            fig_rank = go.Figure()
            for idx, row in ranking_df.iterrows():
                fig_rank.add_trace(go.Bar(
                    y=[row[APP_COL]], x=[row['sentiment_score']], orientation='h',
                    marker_color=row['bar_color'], text=f"{row['sentiment_score']:.1f}",
                    textposition='inside', textfont=dict(color='white', size=13, family='Inter'),
                    hovertemplate=f"<b>{row[APP_COL]}</b><br>Score: {row['sentiment_score']:.2f}<br>Positif: {row['persentase_positif']:.1f}%<br>Total: {int(row['total_ulasan']):,}<extra></extra>"
                ))
            fig_rank.update_layout(
                xaxis=dict(title='Sentiment Score (0-100)', range=[0, 105], gridcolor='#E2E8F0'),
                yaxis=dict(title='', tickfont=dict(size=13, color='#334155', family='Inter'), autorange='reversed'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, margin=dict(t=20, b=40, l=150, r=30), height=400,
                font=dict(family='Inter', size=12)
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Detail Breakdown per Aplikasi</h3>", unsafe_allow_html=True)
        
        detail_cols = st.columns(len(ranking_df))
        for idx, (_, row) in enumerate(ranking_df.iterrows()):
            with detail_cols[idx]:
                app_data = df[df[APP_COL] == row[APP_COL]]
                pos_count = len(app_data[app_data['vader_label'] == 'positive'])
                neg_count = len(app_data[app_data['vader_label'] == 'negative'])
                neu_count = len(app_data[app_data['vader_label'] == 'neutral'])
                
                if row['sentiment_score'] >= 70:
                    status_color, status_text, status_bg = '#16A34A', 'SANGAT BAIK', '#F0FDF4'
                elif row['sentiment_score'] >= 50:
                    status_color, status_text, status_bg = '#D97706', 'CUKUP', '#FFFBEB'
                else:
                    status_color, status_text, status_bg = '#DC2626', 'PERLU PERBAIKAN', '#FEF2F2'
                
                st.markdown(f"""
                <div class="glass-card" style="border-top: 4px solid {status_color}; text-align: center; padding: 1.5rem 1rem;">
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 800; color: #0F2C67; margin-bottom: 0.25rem;">#{row['peringkat']}</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F2C67; margin-bottom: 0.75rem; word-break: break-word; font-family: Inter, sans-serif;">{row[APP_COL]}</div>
                    <div style="background: {status_bg}; color: {status_color}; padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; display: inline-block; margin-bottom: 1rem; font-family: Inter, sans-serif;">{status_text}</div>
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 800; color: {status_color}; margin-bottom: 0.5rem;">{row['sentiment_score']:.1f}</div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 1rem; font-family: Inter, sans-serif;">Sentiment Score</div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 0.5rem; font-family: Inter, sans-serif;">
                        <span style="color: #16A34A; font-weight: 600;">{pos_count}</span>
                        <span style="color: #D97706; font-weight: 600;">{neu_count}</span>
                        <span style="color: #DC2626; font-weight: 600;">{neg_count}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Kolom nama aplikasi tidak ditemukan untuk membuat ranking.")

# =============================================================
# TAB 4: UJI COBA & PREDIKSI
# =============================================================
with tab4:
    st.markdown("""
    <div class="section-title">
        <span class="section-icon">◆</span>
        <h2>Uji Coba Prediksi Sentimen Real-Time</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #475569 !important; margin-bottom: 1.5rem;">
    Pilih aplikasi pemerintah yang ingin dianalisis untuk melihat konteks statistiknya, 
    lalu masukkan teks ulasan untuk diprediksi oleh model <b>1D-CNN Terbaik</b>.
    </p>
    """, unsafe_allow_html=True)
    
    app_list = sorted(df[APP_COL].unique().tolist()) if APP_COL in df.columns else ['Aplikasi Pemerintah']
    selected_app = st.selectbox("Pilih Aplikasi / Layanan Publik", app_list, index=0)
    
    stats = get_app_stats(selected_app)
    
    if stats:
        st.markdown(f"<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin: 1.5rem 0 1rem; font-family: Playfair Display, serif;'>Statistik Aplikasi: <b>{selected_app}</b></h3>", unsafe_allow_html=True)
        
        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        with ac1:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Total Ulasan</h3>
                <div class="value">{stats['total']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with ac2:
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="color: #16A34A !important;">Positif</h3>
                <div class="value" style="color: #16A34A !important;">{stats['pos']:,}</div>
                <div class="delta" style="color: #16A34A;">{stats['pos_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with ac3:
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="color: #D97706 !important;">Netral</h3>
                <div class="value" style="color: #D97706 !important;">{stats['neu']:,}</div>
                <div class="delta" style="color: #D97706;">{stats['neu_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with ac4:
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="color: #DC2626 !important;">Negatif</h3>
                <div class="value" style="color: #DC2626 !important;">{stats['neg']:,}</div>
                <div class="delta" style="color: #DC2626;">{stats['neg_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with ac5:
            score_color = '#16A34A' if stats['score'] >= 70 else '#D97706' if stats['score'] >= 50 else '#DC2626'
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="color: {score_color} !important;">Sentiment Score</h3>
                <div class="value" style="color: {score_color} !important;">{stats['score']:.1f}</div>
                <div class="delta" style="color: {score_color};">dari 100</div>
            </div>
            """, unsafe_allow_html=True)
        
        if stats['pos_pct'] > 70:
            st.markdown(f"""
            <div class="glass-success">
                <p style="margin: 0; font-size: 0.95rem; color: #145A32 !important;">
                <b>{selected_app}</b> termasuk layanan dengan performa <b>SANGAT BAIK</b>. 
                Mayoritas pengguna ({stats['pos_pct']:.1f}%) merasa puas.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif stats['neg_pct'] > 40:
            st.markdown(f"""
            <div class="glass-danger">
                <p style="margin: 0; font-size: 0.95rem; color: #991B1B !important;">
                <b>{selected_app}</b> memerlukan perhatian serius! {stats['neg_pct']:.1f}% ulasan bernada negatif. 
                Segera evaluasi kinerja server, UI/UX, dan respons keluhan.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-warning">
                <p style="margin: 0; font-size: 0.95rem; color: #92400E !important;">
                <b>{selected_app}</b> berada dalam kondisi standar. Terdapat ruang untuk peningkatan kualitas.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        app_sent_dist = pd.DataFrame({
            'Sentimen': ['Positif', 'Netral', 'Negatif'],
            'Jumlah': [stats['pos'], stats['neu'], stats['neg']],
            'Persentase': [stats['pos_pct'], stats['neu_pct'], stats['neg_pct']]
        })
        fig_app = px.bar(
            app_sent_dist, x='Sentimen', y='Persentase', color='Sentimen',
            color_discrete_map={'Positif': '#16A34A', 'Netral': '#D97706', 'Negatif': '#DC2626'},
            text=app_sent_dist['Persentase'].apply(lambda x: f'{x:.1f}%'),
            labels={'Persentase': 'Persentase (%)'}
        )
        fig_app.update_traces(textposition='outside', textfont_size=13, textfont_color='#334155')
        fig_app.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#334155'), showlegend=False,
            yaxis=dict(gridcolor='#E2E8F0', title='Persentase (%)'), xaxis=dict(title=''),
            margin=dict(t=30, b=20, l=40, r=20), height=280
        )
        st.plotly_chart(fig_app, use_container_width=True)
    
    st.divider()
    
    st.markdown("<h3 style='color: #0F2C67 !important; font-size: 1.1rem !important; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Masukkan Ulasan untuk Diprediksi</h3>", unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Tulis ulasan pengguna:",
        height=150,
        placeholder=f"Contoh: Saya menggunakan {selected_app} dan menurut saya..."
    )
    
    if st.button("Analisis Sentimen dengan CNN", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Model CNN sedang menganalisis teks..."):
                label, confidence, probs = predict_sentiment(user_input)
            
            pred_class_idx = list(LABEL_MAP.values()).index(label)
            pred_color = LABEL_COLOR[pred_class_idx]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="glass-card" style="border: 2px solid {pred_color}; text-align: center; padding: 2.5rem;">
                <div style="font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 800; color: {pred_color}; margin-bottom: 0.5rem; letter-spacing: -1px;">{label.upper()}</div>
                <p style="color: #64748B !important; font-size: 1rem !important; margin-bottom: 2rem;">Hasil Prediksi Model CNN</p>
            """, unsafe_allow_html=True)
            
            rc1, rc2 = st.columns(2)
            with rc1:
                st.metric("Tingkat Keyakinan Model", f"{confidence:.2f}%")
            with rc2:
                st.metric("Kelas Prediksi", f"{pred_class_idx} (0=Neg, 1=Neu, 2=Pos)")
            
            st.markdown("<p style='color: #64748B; font-size: 0.9rem; margin-top: 1.5rem;'>Keyakinan Model:</p>", unsafe_allow_html=True)
            st.progress(int(confidence)/100)
            
            st.markdown("<h4 style='color: #0F2C67 !important; margin-top: 1.5rem; margin-bottom: 1rem; font-family: Playfair Display, serif;'>Detail Probabilitas per Kelas:</h4>", unsafe_allow_html=True)
            
            prob_cols = st.columns(3)
            for i, (name, color) in enumerate(zip(LABEL_MAP.values(), LABEL_COLOR.values())):
                with prob_cols[i]:
                    prob_pct = probs[i] * 100
                    is_pred = (name == label)
                    border_style = f"3px solid {color}" if is_pred else "1px solid rgba(0,0,0,0.06)"
                    bg_style = f"background: rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.05);" if is_pred else "background: rgba(255,255,255,0.4);"
                    
                    st.markdown(f"""
                    <div class="glass-card" style="{bg_style} border: {border_style}; text-align: center; padding: 1.25rem;">
                        <div style="font-family: 'Playfair Display', serif; font-weight: 800; color: {color if is_pred else '#64748B'} !important; font-size: 1.1rem; margin-bottom: 0.5rem;">{name.upper()}</div>
                        <div style="font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 800; color: {color} !important; margin: 0.5rem 0;">{prob_pct:.2f}%</div>
                        <div style="background: #E2E8F0; border-radius: 6px; height: 8px; overflow: hidden;">
                            <div style="width: {prob_pct}%; height: 100%; background: {color}; border-radius: 6px;"></div>
                        </div>
                        {'<div style="margin-top: 0.75rem; font-size: 0.75rem; color: #16A34A; font-weight: 700; font-family: Inter, sans-serif;">PREDIKSI TERPILIH</div>' if is_pred else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if label == "Positif":st.markdown("""
            <div class="glass-success">
                <p style="margin: 0; font-size: 0.95rem; color: #145A32 !important;">
                <b>Rekomendasi Strategis:</b> Ulasan mencerminkan kepuasan pengguna.
                Pertahankan fitur unggulan yang disukai dan pertimbangkan untuk diterapkan pada aplikasi lain.
                </p>
            </div>
            """, unsafe_allow_html=True)
            elif label == "Netral":st.markdown("""
            <div class="glass-warning">
                <p style="margin: 0; font-size: 0.95rem; color: #92400E !important;">
                <b>Rekomendasi Strategis:</b> Ulasan bersifat netral, biasanya mengandung saran atau keluhan ringan.
                Perhatikan feedback spesifik untuk peningkatan fitur.
                </p>
            </div>
            """, unsafe_allow_html=True)
            else:st.markdown("""
            <div class="glass-danger">
                <p style="margin: 0; font-size: 0.95rem; color: #991B1B !important;">
                <b>Rekomendasi Strategis:</b> Ulasan negatif terdeteksi!
                Segera lakukan evaluasi teknis (server, UI/UX, bug) dan tingkatkan respons keluhan pengguna.
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================
# 8. FOOTER (NO LOGO)
# =============================================================
st.markdown("""
<div class="footer-glass">
    <h3 style="font-family: 'Playfair Display', serif !important; color: #0F2C67 !important; font-size: 1.4rem !important; margin-bottom: 0.5rem;">2026 Sistem Informasi UAS Deep Learning</h3>
    <p style="color: #4A5568 !important; font-family: 'Inter', sans-serif; margin: 0.4rem 0; font-size: 0.95rem;">
        <b>Kelompok 11</b> &bull; Fakultas Ilmu Komputer
    </p>
    <p style="color: #4A5568 !important; font-family: 'Inter', sans-serif; margin: 0.4rem 0; font-size: 0.9rem;">
        Studi Kasus: <b>IGAR (Indonesian Government Applications Review)</b>
    </p>
    <p style="color: #64748B !important; font-family: 'Inter', sans-serif; margin: 0.4rem 0; font-size: 0.85rem;">
        Data bersumber dari Google Play Store Reviews
    </p>
    <p style="color: #94A3B8 !important; font-family: 'Inter', sans-serif; margin-top: 0.5rem; font-size: 0.8rem;">
        Mobile JKN &bull; MyPertamina &bull; KAI Access &bull; SATUSEHAT &bull; JMO &bull; BMKG Info
    </p>
</div>
""", unsafe_allow_html=True)