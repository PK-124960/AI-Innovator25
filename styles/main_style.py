import streamlit as st

def load_css():
    st.markdown("""
    <style>
        /* --- Import Google Font: Sarabun --- */
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');

        /* --- Global Styles & Earth Tone Theme --- */
        * { font-family: 'Sarabun', sans-serif !important; }
        .stApp { background-color: #F5F5F5; } /* Light Sand Background */
        .main .block-container { padding: 1rem 2rem 3rem 2rem; }

        /* --- Hide Streamlit's default auto-generated menu --- */
        [data-testid="stSidebarNav"] { display: none; }

        /*
        ========================================================================
        ### Sidebar Transformation: Earth Tone & Framed Design ###
        ========================================================================
        */
        [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
        [data-testid="stSidebar"] > div:first-child {
            display: flex; flex-direction: column; height: 100%; padding-bottom: 1rem;
        }
        
        /* --- Sidebar: Branding Area --- */
        .sidebar-branding {
            padding: 2rem 1rem 1.5rem 1rem; text-align: center;
        }
        .sidebar-branding img {
            max-width: 70%; height: auto; object-fit: contain; margin-bottom: 1rem;
            border-radius: 0.75rem;
        }
        .sidebar-branding .title { font-size: 1.2rem; font-weight: 600; color: #4E443A; }

        /* --- Sidebar: Divider Line --- */
        .sidebar-divider {
            margin: 0 1rem 1rem 1rem;
            border-bottom: 1px solid #E0E0E0;
        }
        
        /* --- Sidebar: Custom Navigation with Frames & Earth Tones --- */
        .stPageLink a {
            display: flex; align-items: center; gap: 12px; padding: 0.8rem 1rem;
            border-radius: 0.5rem; text-decoration: none; color: #5D544A;
            font-weight: 500; transition: all 0.2s ease-in-out;
            margin: 0.25rem 1rem; /* ระยะห่างรอบปุ่ม */
            border: 1px solid transparent; /* กรอบใสๆ รอไว้ */
        }
        .stPageLink a:hover {
            background-color: #F5F5F5;
            border: 1px solid #D87355; /* กรอบสี Terracotta เมื่อ hover */
            color: #4E443A;
        }
        .stPageLink a[aria-current="page"] { /* Active page style */
            background-color: #D87355; /* สี Terracotta */
            color: #FFFFFF;
            font-weight: 600;
            border: 1px solid #D87355;
        }

        /* --- Sidebar: Footer & Status --- */
        .sidebar-footer {
            padding: 0 1.5rem 1rem 1.5rem; flex-grow: 1;
            display: flex; flex-direction: column; justify-content: flex-end;
        }
        .status-indicator {
            display: flex; align-items: center; gap: 10px; padding: 0.75rem;
            border-radius: 0.5rem; margin-bottom: 1rem; font-weight: 500;
            border: 1px solid #B2C2A0; /* Sage Green Border */
        }
        .status-indicator.online { background-color: #F1F5E8; color: #515E3D; } /* Sage Green Light */
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background-color: #8A9A5B; }
        .powered-by-box { background-color: #F5F5F5; border-radius: 0.75rem; padding: 1rem; text-align: center; }
        .powered-by-box small { color: #5D544A; display: block; }
        .powered-by-box .brand-name { color: #4E443A; font-weight: 600; text-decoration: none; }

        /*
        ========================================================================
        ### Main Content Transformation: Framed & Colorful ###
        ========================================================================
        */
        .hero-header {
            width: 100%; height: 150px; border-radius: 1rem; background-size: cover;
            background-position: center; margin-bottom: 2rem;
            border: 1px solid #E0E0E0;
        }

        /* --- Main Content Card --- */
        [data-testid="stVerticalBlock"] > [style*="border: 1px solid"] {
            background-color: #FFFFFF;
            border-radius: 1rem;
            padding: 2rem 2.5rem;
            border: 1px solid #E0E0E0 !important;
        }

        /* --- Force Text Color inside the Main Content Area --- */
        .main h1, .main h2, .main h3, .main h4, .main p, .main li, .main .stMarkdown, .main label {
            color: #4E443A !important;
        }
        .main .stAlert p { color: #4E443A !important; }

        /* --- General Headers Styling --- */
        h1, h2, h3 { color: #4E443A; }
        h1 { font-size: 2.2rem; font-weight: 700; display: flex; align-items: center; gap: 1rem; }
        h2 { font-size: 1.6rem; font-weight: 600; }
        h3 { font-size: 1.2rem; font-weight: 600; color: #D87355; }

        /* --- Button Styling with Terracotta Accent --- */
        .stButton>button {
            background-color: #D87355;
            color: #FFFFFF;
            border: none;
            border-radius: 0.5rem;
            padding: 0.8rem 1.5rem;
            font-weight: 600; font-size: 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #C1664A;
            box-shadow: 0 4px 15px rgba(216, 115, 85, 0.3);
            transform: translateY(-2px);
        }
        .stButton>button:disabled {
             background-color: #D3C9C1;
             color: #FFFFFF;
        }

    </style>
    """, unsafe_allow_html=True)