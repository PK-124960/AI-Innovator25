import streamlit as st
from pathlib import Path
from .file_helper import image_to_base64
from utils.llm_helper import init_ollama_client

OLLAMA_AVAILABLE = init_ollama_client()

def render_sidebar():
    # โหลด Assets
    sidebar_logo_b64 = image_to_base64(Path("assets/logo.png"))

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-branding">
                <img src="data:image/png;base64,{sidebar_logo_b64}" />
                <div class="title">ระบบสร้างเอกสารราชการอัตโนมัติ</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # เมนู Navigation
        st.page_link("app.py", label="หน้าแรก", icon="🏠")
        st.page_link("pages/1_✍️_ร่างหนังสือราชการ.py", label="ร่างหนังสือราชการ", icon="✍️")
        st.page_link("pages/2_📬_สร้างหนังสือตอบกลับ.py", label="สร้างหนังสือตอบกลับ", icon="📬")
        st.page_link("pages/3_🤖_ที่ปรึกษาอัจฉริยะ.py", label="ที่ปรึกษาอัจฉริยะ", icon="🤖")
            
        # ส่วน Footer
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True) 
        
        st.markdown(
            """
            <div class="sidebar-footer">
                <div class="status-indicator online">
                    <div class="status-dot"></div>
                    <span>AI Status: {status_text}</span>
                </div>
                <div class="powered-by-box">
                    <small>พัฒนาร่วมกับ</small>
                    <span class="brand-name">RTARF AI Innovator 2025</span>
                    <small>FINE-TUNER</small>
                </div>
            </div>
            """.format(
                status_text="Online" if OLLAMA_AVAILABLE else "Offline"
            ),
            unsafe_allow_html=True
        )

def reset_workflow_states():
    
    st.session_state.ocr_text_content = None
    st.session_state.extracted_data = None
    st.session_state.current_doc_type_for_data = None
    st.session_state.reply_content = ""
    # st.session_state.uploaded_file_name = None # <--- ไม่ต้องรีเซ็ตตัวนี้ที่นี่ เพราะจะถูกตั้งค่าใหม่ทันที
    st.session_state.confirmed_opening_paragraph = ""
    st.session_state.full_reply_draft = ""
    st.session_state.is_draft_generated = False
    st.session_state.opening_options = []
    st.session_state.selected_opening = ""
    # ไม่ต้องล้าง log การแก้ไขก็ได้ เพื่อให้ผู้ใช้ยังเห็น feedback ของตัวเองอยู่
    # st.session_state.opening_corrections_log = [] 
    st.toast("เริ่มต้นกระบวนการสำหรับไฟล์ใหม่", icon="🔄")