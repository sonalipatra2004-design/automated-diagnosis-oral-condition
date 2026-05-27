# ══════════════════════════════════════════════════════
# AUTOMATED DIAGNOSIS OF ORAL CONDITIONS
# Professional Streamlit Web App
# Group 08 | ITER SOA University | MCA 2024-2026
# ══════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import json
import datetime
import hashlib
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="OralDx — Oral Disease Detector",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,
        #0a0e1a 0%, #0d1628 50%, #0a1520 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        #0d1628 0%, #0a1520 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    transition: all 0.3s ease;
}

.card:hover {
    border-color: rgba(0,212,255,0.4);
    background: rgba(0,212,255,0.06);
}

/* Hero section */
.hero {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.12) 0%,
        rgba(0,100,200,0.08) 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00d4ff, #0099cc, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* Team member card */
.member-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
}

.member-card:hover {
    border-color: rgba(0,212,255,0.5);
    background: rgba(0,212,255,0.08);
    transform: translateY(-4px);
}

.member-avatar {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid rgba(0,212,255,0.5);
    margin: 0 auto 0.8rem;
    display: block;
    background: linear-gradient(135deg,#00d4ff,#0066aa);
}

.avatar-placeholder {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: linear-gradient(135deg,#00d4ff,#0066aa);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin: 0 auto 0.8rem;
    border: 3px solid rgba(0,212,255,0.5);
}

.member-name {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.member-reg {
    color: #00d4ff;
    font-size: 0.78rem;
    font-weight: 500;
}

.member-role {
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

/* Result card */
.result-hero {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.12),
        rgba(0,80,180,0.08));
    border: 2px solid rgba(0,212,255,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.disease-big {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Space Grotesk',sans-serif;
}

.confidence-text {
    font-size: 1.1rem;
    color: #94a3b8;
    margin: 0.3rem 0;
}

/* Severity badges */
.badge-mild {
    background: rgba(56,161,105,0.2);
    border: 1px solid #38a169;
    color: #68d391;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-moderate {
    background: rgba(214,158,46,0.2);
    border: 1px solid #d69e2e;
    color: #f6e05e;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-severe {
    background: rgba(229,62,62,0.2);
    border: 1px solid #e53e3e;
    color: #fc8181;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

/* Info rows */
.info-block {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.info-label {
    color: #00d4ff;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.info-value {
    color: #e2e8f0;
    font-size: 0.92rem;
    margin-top: 0.15rem;
}

/* History item */
.history-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.history-item:hover {
    border-color: rgba(0,212,255,0.3);
    background: rgba(0,212,255,0.05);
}

/* Stats */
.stat-card {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Space Grotesk',sans-serif;
}
.stat-label {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-family: 'Space Grotesk',sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,212,255,0.2);
}

/* Auth form */
.auth-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 2.5rem;
    max-width: 420px;
    margin: 2rem auto;
}

/* Input styling */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg,
        #00d4ff, #0099cc) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    font-family: 'Plus Jakarta Sans',sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.35) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

/* Guide card */
.guide-card {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.1),
        rgba(0,80,200,0.05));
    border: 2px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.upload-area {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.page-nav {
    color: #94a3b8;
    font-size: 0.82rem;
    text-align: center;
    margin-bottom: 1rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(0,212,255,0.25);
    border-radius: 12px;
    padding: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════
if 'logged_in'    not in st.session_state:
    st.session_state.logged_in    = False
if 'user_email'   not in st.session_state:
    st.session_state.user_email   = ''
if 'user_name'    not in st.session_state:
    st.session_state.user_name    = ''
if 'history'      not in st.session_state:
    st.session_state.history      = []
if 'page'         not in st.session_state:
    st.session_state.page         = 'home'
if 'users_db'     not in st.session_state:
    st.session_state.users_db     = {}
if 'total_scans'  not in st.session_state:
    st.session_state.total_scans  = 0

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════
CLASSES = [
    'Calculus','Data caries','Gingivitis',
    'Hypodontia','Mouth Ulcer',
    'Tooth Discoloration'
]

DISEASE_INFO = {
    'Calculus':{
        'icon':'🦷',
        'description':'Hardened dental plaque on tooth surfaces.',
        'symptoms':'Yellow/brown deposits, bad breath, gum irritation.',
        'treatment':'Professional dental cleaning (scaling).',
        'severity':'Mild-Moderate',
        'color':'#f6ad55'
    },
    'Data caries':{
        'icon':'⚠️',
        'description':'Dental decay caused by bacterial acid erosion.',
        'symptoms':'Toothache, sensitivity, dark spots on teeth.',
        'treatment':'Fillings, root canal, or extraction if severe.',
        'severity':'Moderate-Severe',
        'color':'#fc8181'
    },
    'Gingivitis':{
        'icon':'🔴',
        'description':'Inflammation of gum tissue around teeth.',
        'symptoms':'Red, swollen, bleeding gums.',
        'treatment':'Improved oral hygiene, professional cleaning.',
        'severity':'Mild',
        'color':'#68d391'
    },
    'Hypodontia':{
        'icon':'❓',
        'description':'Congenital absence of one or more teeth.',
        'symptoms':'Missing teeth, gaps in dental arch.',
        'treatment':'Dental implants, bridges, or dentures.',
        'severity':'Moderate',
        'color':'#b794f4'
    },
    'Mouth Ulcer':{
        'icon':'💊',
        'description':'Painful sores on soft tissues in the mouth.',
        'symptoms':'Painful lesions, difficulty eating or speaking.',
        'treatment':'Topical gel, antiseptic mouthwash.',
        'severity':'Mild',
        'color':'#4fd1c5'
    },
    'Tooth Discoloration':{
        'icon':'🌟',
        'description':'Visible colour change in teeth due to staining.',
        'symptoms':'Yellow, brown, or black discolouration on teeth.',
        'treatment':'Whitening, veneers, or restorative procedures.',
        'severity':'Mild',
        'color':'#f6e05e'
    },
}

TEAM_MEMBERS = [
    {
        'name'   : 'Sonali Patra',
        'reg'    : '24C216A45',
        'role'   : 'Team Lead & ML Engineer',
        'emoji'  : '👩‍💻',
        'contrib': 'Model training, Streamlit app development'
    },
    {
        'name'   : 'Jagruti Parida',
        'reg'    : '24C216A42',
        'role'   : 'Data Engineer',
        'emoji'  : '👩‍🔬',
        'contrib': 'Dataset collection and preprocessing'
    },
    {
        'name'   : 'Dharitri Pradhan',
        'reg'    : '24C216A30',
        'role'   : 'Deep Learning Researcher',
        'emoji'  : '👩‍💡',
        'contrib': 'CNN architecture design and evaluation'
    },
    {
        'name'   : 'Smitarani Mohapatra',
        'reg'    : '24C213A05',
        'role'   : 'Frontend Developer',
        'emoji'  : '👩‍🎨',
        'contrib': 'UI/UX design and documentation'
    },
    {
        'name'   : 'Barsha Priyadarshini Singh',
        'reg'    : '24C219A30',
        'role'   : 'Backend Developer',
        'emoji'  : '👩‍🏫',
        'contrib': 'Model deployment and testing'
    },
]

GUIDE = {
    'name'       : 'Dr. Debabrata Singh',
    'designation': 'Associate Professor',
    'dept'       : 'Department of Computer Application',
    'university' : 'ITER, SOA University, Bhubaneswar',
    'emoji'      : '👨‍🏫'
}

# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def hash_password(password):
    return hashlib.sha256(
        password.encode()).hexdigest()

def preprocess(img_array):
    r     = cv2.resize(img_array,(224,224))
    lab   = cv2.cvtColor(
        r, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enh   = cv2.cvtColor(
        lab, cv2.COLOR_LAB2RGB)
    den   = cv2.GaussianBlur(enh,(3,3),0)
    norm  = den.astype(np.float32)/255.0
    return norm, r, enh, den

def get_severity_badge(severity):
    if 'Severe' in severity and 'Mild' not in severity:
        return f'<span class="badge-severe">{severity}</span>'
    elif 'Moderate' in severity:
        return f'<span class="badge-moderate">{severity}</span>'
    else:
        return f'<span class="badge-mild">{severity}</span>'

def img_to_base64(img_array):
    pil_img = Image.fromarray(
        img_array.astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(
        buf.getvalue()).decode()

@st.cache_resource
def load_model():
    paths = [
        'vgg19_best.keras',
        'resnet50_best.keras',
        '/content/drive/MyDrive/OralDiagnosis_Results/vgg19_best.keras',
        '/content/drive/MyDrive/OralDiagnosis_Results/resnet50_best.keras',
    ]
    for path in paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path
    return None, None

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:2.5rem;">🦷</div>
        <div style="color:#00d4ff;font-weight:800;
                    font-size:1.2rem;font-family:
                    'Space Grotesk',sans-serif;">
            OralDx
        </div>
        <div style="color:#475569;font-size:0.72rem;">
            Oral Disease Detector
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.logged_in:
        st.markdown(f"""
        <div style="background:rgba(0,212,255,0.08);
                    border:1px solid rgba(0,212,255,0.2);
                    border-radius:12px;padding:0.8rem;
                    margin-bottom:1rem;text-align:center;">
            <div style="font-size:1.5rem;">👤</div>
            <div style="color:#e2e8f0;font-weight:600;
                        font-size:0.9rem;">
                {st.session_state.user_name}
            </div>
            <div style="color:#475569;font-size:0.75rem;">
                {st.session_state.user_email}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation
    pages = {
        '🏠 Home'     : 'home',
        '🔍 Diagnose' : 'diagnose',
        '📋 History'  : 'history',
        '👥 About Us' : 'about',
        'ℹ️ Info'     : 'info',
    }

    st.markdown(
        '<div style="color:#475569;font-size:0.75rem;'
        'font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:0.5rem;">'
        'NAVIGATION</div>',
        unsafe_allow_html=True)

    for label, page_key in pages.items():
        active = (st.session_state.page
                  == page_key)
        bg = ("rgba(0,212,255,0.12)"
              if active else "transparent")
        border = ("rgba(0,212,255,0.4)"
                  if active else "transparent")
        if st.button(label, key=f"nav_{page_key}"):
            st.session_state.page = page_key
            st.rerun()

    st.markdown("---")

    if st.session_state.logged_in:
        if st.button("🚪 Sign Out"):
            st.session_state.logged_in = False
            st.session_state.user_email = ''
            st.session_state.user_name  = ''
            st.session_state.page = 'home'
            st.rerun()
    else:
        if st.button("🔐 Sign In / Sign Up"):
            st.session_state.page = 'auth'
            st.rerun()

    # Stats
    st.markdown("---")
    st.markdown("""
    <div style="color:#475569;font-size:0.72rem;
                text-align:center;">
        MCA Final Year Project<br>
        Group 08 | Batch 2024-2026<br>
        ITER, SOA University
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: AUTH
# ══════════════════════════════════════════════════════
def page_auth():
    st.markdown("""
    <div style="text-align:center;margin:1rem 0 2rem;">
        <div style="font-size:3rem;">🦷</div>
        <h2 style="color:#00d4ff;font-family:
                   'Space Grotesk',sans-serif;
                   margin:0.3rem 0;">
            Welcome to OralDx
        </h2>
        <p style="color:#475569;">
            Sign in to access all features
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(
            ["🔐 Sign In", "📝 Sign Up"])

        with tab1:
            st.markdown("#### Sign In")
            email = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                key="signin_email")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                key="signin_pass")
            if st.button("Sign In →",
                         key="do_signin"):
                if email and password:
                    users = st.session_state.users_db
                    hp = hash_password(password)
                    if (email in users and
                            users[email]['password']
                            == hp):
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.user_name  = \
                            users[email]['name']
                        st.session_state.page = 'home'
                        st.success("Signed in!")
                        st.rerun()
                    else:
                        st.error(
                            "Invalid email or password")
                else:
                    st.warning("Fill all fields")

        with tab2:
            st.markdown("#### Create Account")
            name = st.text_input(
                "Full Name",
                placeholder="Dr. John Smith",
                key="signup_name")
            email2 = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                key="signup_email")
            pass2 = st.text_input(
                "Password",
                type="password",
                placeholder="Min 6 characters",
                key="signup_pass")
            role = st.selectbox(
                "I am a...",
                ["Dental Professional",
                 "Medical Student",
                 "Researcher",
                 "General User"])
            if st.button("Create Account →",
                         key="do_signup"):
                if name and email2 and pass2:
                    if len(pass2) < 6:
                        st.error(
                            "Password min 6 characters")
                    elif email2 in \
                            st.session_state.users_db:
                        st.error("Email already registered")
                    else:
                        st.session_state.users_db[
                            email2] = {
                            'name'    : name,
                            'password': hash_password(pass2),
                            'role'    : role,
                            'joined'  : str(
                                datetime.date.today())
                        }
                        st.session_state.logged_in = True
                        st.session_state.user_email = email2
                        st.session_state.user_name  = name
                        st.session_state.page = 'home'
                        st.success("Account created!")
                        st.rerun()
                else:
                    st.warning("Fill all fields")

        # Demo login
        st.markdown("---")
        st.markdown(
            '<p style="color:#475569;text-align:center;'
            'font-size:0.82rem;">Try demo account</p>',
            unsafe_allow_html=True)
        if st.button("🚀 Continue as Guest"):
            st.session_state.logged_in  = True
            st.session_state.user_email = "guest@oraldx.app"
            st.session_state.user_name  = "Guest User"
            st.session_state.page       = 'diagnose'
            st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════
def page_home():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🦷 OralDx</h1>
        <h3 style="color:#e2e8f0;font-weight:500;
                   font-size:1.2rem;margin:0.5rem 0;">
            Automated Diagnosis of Oral Conditions
            from Dental X-Rays
        </h3>
        <p>AI-powered dental disease detection using
           Deep Learning | VGG19 · ResNet-50 · YOLOv8
           · U-Net</p>
        <p style="color:#475569;font-size:0.85rem;
                  margin-top:0.5rem;">
            Department of Computer Application |
            ITER, SOA University |
            MCA Batch 2024-2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        ("91%",  "Best Accuracy",     "VGG19 Model"),
        ("6",    "Diseases",          "Detected"),
        ("12K+", "Training Images",   "Dataset Size"),
        (str(st.session_state.total_scans),
         "Total Scans", "All Users"),
    ]
    for col, (num, label, sub) in zip(
            [c1,c2,c3,c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{num}</div>
                <div style="color:#e2e8f0;
                            font-weight:600;
                            font-size:0.85rem;">
                    {label}
                </div>
                <div class="stat-label">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown(
        '<div class="section-title">'
        '✨ Features</div>',
        unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    features = [
        ("🔍","Instant Diagnosis",
         "Upload any dental X-ray and get instant disease detection with confidence score"),
        ("📊","Detailed Analysis",
         "Preprocessing pipeline, probability charts, and detailed disease information"),
        ("📋","History Tracking",
         "All your previous diagnoses saved and accessible anytime"),
        ("🔐","Secure Login",
         "Create account to save your history and access advanced features"),
        ("📱","Mobile Friendly",
         "Works perfectly on phones, tablets and laptops"),
        ("🎯","91% Accuracy",
         "Trained on 12,320+ dental X-ray images across 6 disease categories"),
    ]
    for i, col in enumerate([f1,f2,f3]):
        with col:
            icon,title,desc = features[i]
            st.markdown(f"""
            <div class="card">
                <div style="font-size:2rem;
                            margin-bottom:0.5rem;">
                    {icon}
                </div>
                <div style="color:#e2e8f0;
                            font-weight:700;
                            font-size:0.95rem;
                            margin-bottom:0.3rem;">
                    {title}
                </div>
                <div style="color:#64748b;
                            font-size:0.82rem;
                            line-height:1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f4,f5,f6 = st.columns(3)
    for i, col in enumerate([f4,f5,f6]):
        with col:
            icon,title,desc = features[i+3]
            st.markdown(f"""
            <div class="card">
                <div style="font-size:2rem;
                            margin-bottom:0.5rem;">
                    {icon}
                </div>
                <div style="color:#e2e8f0;
                            font-weight:700;
                            font-size:0.95rem;
                            margin-bottom:0.3rem;">
                    {title}
                </div>
                <div style="color:#64748b;
                            font-size:0.82rem;
                            line-height:1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Diseases
    st.markdown(
        '<div class="section-title">'
        '🦷 Diseases We Detect</div>',
        unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (cls, info) in enumerate(
            DISEASE_INFO.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;
                            align-items:center;
                            gap:0.6rem;
                            margin-bottom:0.5rem;">
                    <span style="font-size:1.5rem;">
                        {info['icon']}
                    </span>
                    <span style="color:#e2e8f0;
                                 font-weight:700;
                                 font-size:0.9rem;">
                        {cls}
                    </span>
                </div>
                <div style="color:#64748b;
                            font-size:0.78rem;
                            line-height:1.5;">
                    {info['description']}
                </div>
                <div style="margin-top:0.5rem;">
                    {get_severity_badge(info['severity'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button(
                "🔍 Start Diagnosis Now →",
                key="home_diagnose"):
            st.session_state.page = 'diagnose'
            st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: DIAGNOSE
# ══════════════════════════════════════════════════════
def page_diagnose():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            🔍 Upload & Diagnose
        </h2>
        <p style="color:#475569;margin:0.3rem 0 0;">
            Upload a dental X-ray for instant
            AI-powered diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning(
            "⚠️ Sign in to save diagnosis history. "
            "You can still use diagnosis as guest.")

    model, model_path = load_model()
    if model is None:
        st.error(
            "Model file not found. Please run "
            "training first (Cell 1 in Colab) and "
            "place vgg19_best.keras in app folder.")
        st.info(
            "Place vgg19_best.keras in same "
            "folder as app.py")
        return

    mname = ("VGG19"
             if "vgg19" in model_path
             else "ResNet-50")
    st.success(
        f"✅ {mname} model loaded — 91% accuracy")

    uploaded_file = st.file_uploader(
        "Choose dental X-ray image",
        type=["jpg","jpeg","png"],
        help="Upload JPG or PNG dental X-ray")

    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert("RGB"))

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(
                '<div class="section-title" '
                'style="font-size:1rem;">'
                '📷 Uploaded Image</div>',
                unsafe_allow_html=True)
            st.image(img,
                     caption=uploaded_file.name,
                     use_column_width=True)

        with col2:
            st.markdown(
                '<div class="section-title" '
                'style="font-size:1rem;">'
                '📋 Image Details</div>',
                unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
                <div class="info-block">
                    <div class="info-label">
                        File Name
                    </div>
                    <div class="info-value">
                        {uploaded_file.name}
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        Dimensions
                    </div>
                    <div class="info-value">
                        {img.width} × {img.height} px
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        Size
                    </div>
                    <div class="info-value">
                        {uploaded_file.size//1024} KB
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        AI Model
                    </div>
                    <div class="info-value">
                        {mname} (91% accuracy)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Run Diagnosis",
                     key="run_diag"):
            with st.spinner(
                    "Analysing X-ray..."):
                norm,resized,enhanced,denoised=\
                    preprocess(img_array)
                probs = model.predict(
                    np.expand_dims(norm,axis=0),
                    verbose=0)[0]
                idx      = np.argmax(probs)
                cls_name = CLASSES[idx]
                conf     = probs[idx]
                info     = DISEASE_INFO[cls_name]
                ts = datetime.datetime.now()\
                    .strftime("%d %b %Y, %H:%M")

            # Save to history
            st.session_state.history.append({
                'disease'   : cls_name,
                'confidence': f'{conf*100:.1f}%',
                'timestamp' : ts,
                'filename'  : uploaded_file.name,
                'severity'  : info['severity'],
                'icon'      : info['icon'],
            })
            st.session_state.total_scans += 1

            # Result
            st.markdown(
                '<div class="section-title">'
                '📋 Diagnosis Result</div>',
                unsafe_allow_html=True)

            sev_badge = get_severity_badge(
                info['severity'])
            st.markdown(f"""
            <div class="result-hero">
                <div class="disease-big">
                    {info['icon']} {cls_name}
                </div>
                <div class="confidence-text">
                    Confidence: 
                    <strong style="color:#00d4ff;">
                        {conf*100:.1f}%
                    </strong>
                </div>
                <div style="margin:0.5rem 0;">
                    {sev_badge}
                </div>
                <div style="color:#475569;
                            font-size:0.78rem;
                            margin-top:0.5rem;">
                    Diagnosed on {ts}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="info-block">
                    <div class="info-label">
                        📝 Description
                    </div>
                    <div class="info-value">
                        {info['description']}
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        🔍 Symptoms
                    </div>
                    <div class="info-value">
                        {info['symptoms']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-block">
                    <div class="info-label">
                        💊 Treatment
                    </div>
                    <div class="info-value">
                        {info['treatment']}
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">
                        ⚠️ Severity
                    </div>
                    <div class="info-value">
                        {info['severity']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Pipeline
            st.markdown(
                '<div class="section-title">'
                '🔬 Preprocessing Pipeline</div>',
                unsafe_allow_html=True)
            stages = [img_array,resized,
                      enhanced,denoised,
                      (norm*255).astype(np.uint8)]
            titles = ['Original','Resized 224×224',
                      'CLAHE','Denoised',
                      'Normalised']
            pipe_cols = st.columns(5)
            for c,st_img,t in zip(
                    pipe_cols,stages,titles):
                with c:
                    st.image(st_img,
                             caption=t,
                             use_column_width=True)

            # Probability chart
            st.markdown(
                '<div class="section-title">'
                '📊 Disease Probabilities</div>',
                unsafe_allow_html=True)

            sorted_p = sorted(
                zip(CLASSES,probs),
                key=lambda x:x[1],
                reverse=True)

            fig, ax = plt.subplots(
                figsize=(10,4),
                facecolor='none')
            ax.set_facecolor(
                'rgba(0,0,0,0)')
            s_cls  = [p[0] for p in sorted_p]
            s_prob = [p[1] for p in sorted_p]
            colors = ['#00d4ff' if c==cls_name
                      else '#1e3a5f'
                      for c in s_cls]
            bars = ax.barh(
                s_cls,s_prob,
                color=colors,
                edgecolor='none',
                height=0.6)
            for bar,prob in zip(bars,s_prob):
                ax.text(
                    bar.get_width()+0.01,
                    bar.get_y()+bar.get_height()/2,
                    f'{prob*100:.1f}%',
                    va='center',
                    fontsize=10,
                    color='white',
                    fontweight='bold')
            ax.set_xlim(0,1.25)
            ax.set_xlabel(
                'Probability',color='white')
            ax.tick_params(
                colors='white',labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(
                '#334155')
            ax.spines['left'].set_color(
                '#334155')
            plt.tight_layout()
            st.pyplot(fig,
                      transparent=True)

            # Disclaimer
            st.markdown("""
            <div style="background:rgba(255,200,0,0.06);
                        border:1px solid rgba(255,200,0,0.2);
                        border-radius:10px;
                        padding:0.8rem 1rem;
                        margin-top:1rem;">
                <small style="color:#fbbf24;">
                    ⚠️ <strong>Disclaimer:</strong>
                    This AI tool is for educational
                    purposes only. Always consult a
                    qualified dental professional for
                    clinical diagnosis and treatment.
                </small>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

# ══════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            📋 Diagnosis History
        </h2>
        <p style="color:#475569;margin:0.3rem 0 0;">
            All your previous diagnoses
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning(
            "Sign in to save and view history.")
        if st.button("🔐 Sign In"):
            st.session_state.page = 'auth'
            st.rerun()
        return

    history = st.session_state.history
    if not history:
        st.markdown("""
        <div class="card" style="text-align:center;
                                  padding:3rem;">
            <div style="font-size:3rem;">📭</div>
            <div style="color:#475569;
                        margin-top:1rem;">
                No diagnoses yet.<br>
                Run your first diagnosis!
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Go to Diagnose"):
            st.session_state.page = 'diagnose'
            st.rerun()
        return

    # Summary stats
    total = len(history)
    diseases = {}
    for h in history:
        d = h['disease']
        diseases[d] = diseases.get(d,0) + 1
    most_common = max(diseases,
                      key=diseases.get)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div style="color:#e2e8f0;
                        font-weight:600;
                        font-size:0.85rem;">
                Total Diagnoses
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">
                {len(diseases)}
            </div>
            <div style="color:#e2e8f0;
                        font-weight:600;
                        font-size:0.85rem;">
                Unique Diseases
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:1.3rem;
                        font-weight:800;
                        color:#00d4ff;">
                {DISEASE_INFO[most_common]['icon']}
                {most_common}
            </div>
            <div style="color:#e2e8f0;
                        font-weight:600;
                        font-size:0.85rem;">
                Most Common
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        '📋 All Records</div>',
        unsafe_allow_html=True)

    # History list (newest first)
    for item in reversed(history):
        info = DISEASE_INFO[item['disease']]
        sev  = item['severity']
        if 'Severe' in sev and 'Mild' not in sev:
            sev_color = '#fc8181'
        elif 'Moderate' in sev:
            sev_color = '#f6e05e'
        else:
            sev_color = '#68d391'

        st.markdown(f"""
        <div class="card" style="display:flex;
                                  align-items:center;
                                  gap:1.2rem;">
            <div style="font-size:2rem;
                        min-width:40px;
                        text-align:center;">
                {info['icon']}
            </div>
            <div style="flex:1;">
                <div style="color:#e2e8f0;
                            font-weight:700;
                            font-size:0.95rem;">
                    {item['disease']}
                </div>
                <div style="color:#475569;
                            font-size:0.78rem;
                            margin-top:0.1rem;">
                    📁 {item['filename']} &nbsp;|&nbsp;
                    🕐 {item['timestamp']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="color:#00d4ff;
                            font-weight:700;
                            font-size:1rem;">
                    {item['confidence']}
                </div>
                <div style="color:{sev_color};
                            font-size:0.75rem;
                            margin-top:0.2rem;">
                    {sev}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ══════════════════════════════════════════════════════
# PAGE: ABOUT US
# ══════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            👥 About Us
        </h2>
        <p style="color:#475569;margin:0.3rem 0 0;">
            Meet the team behind OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project Guide
    st.markdown(
        '<div class="section-title">'
        '🎓 Project Guide</div>',
        unsafe_allow_html=True)

    st.markdown(f"""
    <div class="guide-card">
        <div style="font-size:4rem;
                    margin-bottom:0.8rem;">
            {GUIDE['emoji']}
        </div>
        <div style="font-family:'Space Grotesk',
                    sans-serif;font-size:1.5rem;
                    font-weight:700;color:#e2e8f0;
                    margin-bottom:0.3rem;">
            {GUIDE['name']}
        </div>
        <div style="color:#00d4ff;font-weight:600;
                    font-size:0.95rem;
                    margin-bottom:0.2rem;">
            {GUIDE['designation']}
        </div>
        <div style="color:#475569;
                    font-size:0.85rem;">
            {GUIDE['dept']}<br>
            {GUIDE['university']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Team Members
    st.markdown(
        '<div class="section-title">'
        '👩‍💻 Team Members</div>',
        unsafe_allow_html=True)

    st.info(
        "💡 Upload your photos to replace "
        "the emoji avatars! Place photos named: "
        "sonali.jpg, jagruti.jpg, dharitri.jpg, "
        "smitarani.jpg, barsha.jpg in app folder.")

    # Row 1 — first 3 members
    r1cols = st.columns(3)
    for i, col in enumerate(r1cols):
        m = TEAM_MEMBERS[i]
        photo_names = [
            'sonali.jpg','jagruti.jpg',
            'dharitri.jpg','smitarani.jpg',
            'barsha.jpg']
        photo_path = photo_names[i]
        with col:
            if os.path.exists(photo_path):
                pil_img = Image.open(photo_path)
                pil_img = pil_img.resize((200,200))
                buf = BytesIO()
                pil_img.save(buf, format='PNG')
                b64 = base64.b64encode(
                    buf.getvalue()).decode()
                photo_html = (
                    f'<img src="data:image/png;'
                    f'base64,{b64}" '
                    f'class="member-avatar">')
            else:
                photo_html = (
                    f'<div class="avatar-placeholder">'
                    f'{m["emoji"]}</div>')

            st.markdown(f"""
            <div class="member-card">
                {photo_html}
                <div class="member-name">
                    {m['name']}
                </div>
                <div class="member-reg">
                    Reg: {m['reg']}
                </div>
                <div class="member-role">
                    {m['role']}
                </div>
                <div style="color:#334155;
                            font-size:0.72rem;
                            margin-top:0.4rem;
                            line-height:1.4;">
                    {m['contrib']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        "<br>", unsafe_allow_html=True)

    # Row 2 — last 2 members (centered)
    _,c1,c2,_ = st.columns([0.5,1,1,0.5])
    for i, col in enumerate([c1,c2]):
        m = TEAM_MEMBERS[i+3]
        photo_names = [
            'sonali.jpg','jagruti.jpg',
            'dharitri.jpg','smitarani.jpg',
            'barsha.jpg']
        photo_path = photo_names[i+3]
        with col:
            if os.path.exists(photo_path):
                pil_img = Image.open(photo_path)
                pil_img = pil_img.resize((200,200))
                buf = BytesIO()
                pil_img.save(buf, format='PNG')
                b64 = base64.b64encode(
                    buf.getvalue()).decode()
                photo_html = (
                    f'<img src="data:image/png;'
                    f'base64,{b64}" '
                    f'class="member-avatar">')
            else:
                photo_html = (
                    f'<div class="avatar-placeholder">'
                    f'{m["emoji"]}</div>')

            st.markdown(f"""
            <div class="member-card">
                {photo_html}
                <div class="member-name">
                    {m['name']}
                </div>
                <div class="member-reg">
                    Reg: {m['reg']}
                </div>
                <div class="member-role">
                    {m['role']}
                </div>
                <div style="color:#334155;
                            font-size:0.72rem;
                            margin-top:0.4rem;
                            line-height:1.4;">
                    {m['contrib']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # University info
    st.markdown(
        '<div class="section-title">'
        '🏛️ About the Institution</div>',
        unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="display:flex;gap:1.5rem;
                    align-items:flex-start;">
            <div style="font-size:3rem;">🎓</div>
            <div>
                <div style="color:#e2e8f0;
                            font-weight:700;
                            font-size:1rem;
                            margin-bottom:0.3rem;">
                    Institute of Technical Education
                    and Research (ITER)
                </div>
                <div style="color:#00d4ff;
                            font-size:0.85rem;
                            margin-bottom:0.5rem;">
                    Siksha O Anusandhan (SOA) Deemed
                    to be University
                </div>
                <div style="color:#64748b;
                            font-size:0.82rem;
                            line-height:1.7;">
                    📍 Bhubaneswar, Odisha, India<br>
                    🎓 Department of Computer Application<br>
                    📚 Master in Computer Application
                    (MCA) 2024-2026<br>
                    👥 Group 08 | Final Year Project
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: INFO
# ══════════════════════════════════════════════════════
def page_info():
    st.markdown("""
    <div class="hero" style="padding:1.5rem;">
        <h2 style="font-family:'Space Grotesk',
                   sans-serif;color:#00d4ff;
                   font-size:1.8rem;margin:0;">
            ℹ️ Project Information
        </h2>
        <p style="color:#475569;margin:0.3rem 0 0;">
            Technical details about OralDx
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model Performance
    st.markdown(
        '<div class="section-title">'
        '📊 Model Performance</div>',
        unsafe_allow_html=True)

    models_data = [
        ("VGG19",    "91%","0.91","0.88","0.895","Best performer","#00d4ff"),
        ("ResNet-50","87%","0.87","0.85","0.860","Transfer learning","#68d391"),
        ("YOLOv8",   "88%","0.87","0.84","0.855","Object detection","#f6ad55"),
        ("U-Net",    "85%","0.83","0.82","0.825","Segmentation","#b794f4"),
    ]

    cols = st.columns(4)
    for col,(name,acc,prec,rec,f1,role,clr) in \
            zip(cols, models_data):
        with col:
            st.markdown(f"""
            <div class="card"
                 style="border-color:{clr}33;
                        text-align:center;">
                <div style="color:{clr};
                            font-weight:800;
                            font-size:1.1rem;
                            font-family:'Space Grotesk',
                            sans-serif;">
                    {name}
                </div>
                <div style="font-size:2rem;
                            font-weight:800;
                            color:{clr};
                            margin:0.3rem 0;">
                    {acc}
                </div>
                <div style="color:#64748b;
                            font-size:0.75rem;">
                    Accuracy
                </div>
                <div style="margin-top:0.8rem;
                            font-size:0.75rem;">
                    <div style="display:flex;
                                justify-content:space-between;
                                color:#94a3b8;
                                padding:2px 0;">
                        <span>Precision</span>
                        <span style="color:{clr};">
                            {prec}
                        </span>
                    </div>
                    <div style="display:flex;
                                justify-content:space-between;
                                color:#94a3b8;
                                padding:2px 0;">
                        <span>Recall</span>
                        <span style="color:{clr};">
                            {rec}
                        </span>
                    </div>
                    <div style="display:flex;
                                justify-content:space-between;
                                color:#94a3b8;
                                padding:2px 0;">
                        <span>F1-Score</span>
                        <span style="color:{clr};">
                            {f1}
                        </span>
                    </div>
                </div>
                <div style="margin-top:0.8rem;
                            background:{clr}18;
                            border-radius:6px;
                            padding:3px 8px;
                            color:{clr};
                            font-size:0.72rem;
                            font-weight:600;">
                    {role}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Dataset
    st.markdown(
        '<div class="section-title">'
        '🗃️ Dataset Information</div>',
        unsafe_allow_html=True)

    d_data = [
        ("Calculus",            "1,296","#2ECC71"),
        ("Caries",              "2,601","#E74C3C"),
        ("Gingivitis",          "2,349","#3498DB"),
        ("Hypodontia",          "1,251","#F39C12"),
        ("Mouth Ulcer",         "2,806","#9B59B6"),
        ("Tooth Discoloration", "2,017","#1ABC9C"),
    ]

    st.markdown("""
    <div class="card">
        <div style="display:grid;
                    grid-template-columns:repeat(3,1fr);
                    gap:0.8rem;">
    """, unsafe_allow_html=True)

    for cls,count,clr in d_data:
        info = DISEASE_INFO[cls if cls != "Caries"
                            else "Data caries"]
        st.markdown(f"""
        <div style="background:{clr}11;
                    border:1px solid {clr}44;
                    border-radius:10px;
                    padding:0.8rem;
                    text-align:center;">
            <div style="font-size:1.5rem;">
                {DISEASE_INFO.get(cls, info)['icon']
                 if cls in DISEASE_INFO
                 else info['icon']}
            </div>
            <div style="color:#e2e8f0;
                        font-weight:600;
                        font-size:0.82rem;
                        margin:0.2rem 0;">
                {cls}
            </div>
            <div style="color:{clr};
                        font-weight:700;
                        font-size:1.1rem;">
                {count}
            </div>
            <div style="color:#475569;
                        font-size:0.7rem;">
                images
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True)

    # Total
    st.markdown("""
    <div style="background:rgba(0,212,255,0.08);
                border:1px solid rgba(0,212,255,0.2);
                border-radius:12px;
                padding:1rem;text-align:center;
                margin:0.5rem 0;">
        <span style="color:#00d4ff;font-weight:800;
                     font-size:1.5rem;">12,320</span>
        <span style="color:#94a3b8;font-size:0.9rem;">
             total images across 6 classes
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown(
        '<div class="section-title">'
        '⚙️ Technology Stack</div>',
        unsafe_allow_html=True)

    tech = [
        ("🐍","Python 3.10",
         "Core programming language"),
        ("🧠","TensorFlow / Keras",
         "Deep learning framework"),
        ("👁️","OpenCV",
         "Image preprocessing (CLAHE, denoising)"),
        ("📊","Matplotlib / Seaborn",
         "Data visualization and charts"),
        ("🌐","Streamlit",
         "Web application framework"),
        ("🎯","Transfer Learning",
         "ImageNet pretrained weights"),
    ]
    t1,t2,t3 = st.columns(3)
    for i,(em,name,desc) in enumerate(tech):
        with [t1,t2,t3][i%3]:
            st.markdown(f"""
            <div class="card"
                 style="padding:1rem;">
                <div style="display:flex;
                            align-items:center;
                            gap:0.6rem;">
                    <span style="font-size:1.4rem;">
                        {em}
                    </span>
                    <div>
                        <div style="color:#e2e8f0;
                                    font-weight:600;
                                    font-size:0.85rem;">
                            {name}
                        </div>
                        <div style="color:#475569;
                                    font-size:0.75rem;">
                            {desc}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════
def show_footer():
    st.markdown("""
    <div class="footer">
        <div style="color:#334155;
                    margin-bottom:0.5rem;">
            🦷 OralDx — Automated Diagnosis of
            Oral Conditions from Dental X-Rays
        </div>
        <div>
            Group 08 &nbsp;|&nbsp;
            Department of Computer Application
            &nbsp;|&nbsp;
            ITER SOA University,
            Bhubaneswar &nbsp;|&nbsp;
            MCA Batch 2024-2026
        </div>
        <div style="margin-top:0.3rem;">
            Developed by: Sonali Patra ·
            Jagruti Parida · Dharitri Pradhan ·
            Smitarani Mohapatra ·
            Barsha Priyadarshini Singh
        </div>
        <div style="margin-top:0.3rem;">
            Guide: Dr. Debabrata Singh,
            Associate Professor
        </div>
        <div style="color:#1e293b;
                    margin-top:0.5rem;
                    font-size:0.7rem;">
            ⚠️ For educational purposes only.
            Not a substitute for professional
            dental advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# ROUTER — Show correct page
# ══════════════════════════════════════════════════════
page = st.session_state.page

if page == 'home':
    page_home()
elif page == 'diagnose':
    page_diagnose()
elif page == 'history':
    page_history()
elif page == 'about':
    page_about()
elif page == 'info':
    page_info()
elif page == 'auth':
    page_auth()
else:
    page_home()

show_footer()
