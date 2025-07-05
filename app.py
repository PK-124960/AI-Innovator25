import streamlit as st
from pathlib import Path
import ollama

from styles.main_style import load_css
from utils.file_helper import image_to_base64
from utils.ui_helper import render_sidebar
from utils.ingest_knowledge_base import initialize_knowledge_base

st.set_page_config(
    page_title="ระบบสร้างเอกสารราชการอัตโนมัติ",
    page_icon="🏠",
    layout="wide",
)

load_css()
render_sidebar()

@st.cache_resource
def run_knowledge_base_initialization():
    initialize_knowledge_base()
    return True

KNOWLEDGE_BASE_READY = run_knowledge_base_initialization()

# --- MAIN PAGE CONTENT ---
st.title("ระบบสร้างเอกสารราชการอัตโนมัติ")
st.header("Automatic Official Document Generation System")
st.markdown("---")

# Hero Banner
# rtarf_banner_b64 = image_to_base64(Path("assets/rtarf_banner.png"))
# if rtarf_banner_b64:
#     st.markdown(
#         f'<div class="hero-header" style="background-image: url(data:image/png;base64,{rtarf_banner_b64})"></div>',
#         unsafe_allow_html=True
#     )

st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

with st.container(border=True):
    st.header("ปฏิวัติการทำงานเอกสารสู่ยุคดิจิทัล")
    st.markdown("""
    ระบบสร้างเอกสารราชการอัจฉริยะนี้ ถูกสร้างขึ้นเพื่อเป็นเครื่องมืออันทรงพลังสำหรับเจ้าหน้าที่ราชการไทยทุกคน
    โดยมีเป้าหมายเพื่อลดภาระงานด้านเอกสารที่ซ้ำซ้อน, เพิ่มความเร็วและความแม่นยำ, และปลดปล่อยศักยภาพของท่าน
    ให้มุ่งเน้นไปที่งานเชิงกลยุทธ์ที่สร้างคุณค่าให้แก่องค์กรและประเทศชาติต่อไป
    """)

st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# main content
st.subheader("วัตถุประสงค์หลัก")
st.markdown("""
- **ลดเวลา:** แปลงความคิดจากภาษาพูดให้เป็นเอกสารราชการที่ถูกต้องในไม่กี่วินาที
- **เพิ่มความแม่นยำ:** ลดความผิดพลาดจากการพิมพ์และใช้รูปแบบภาษาที่สอดคล้องกับระเบียบปฏิบัติ
- **เสริมสร้างประสิทธิภาพ:** จัดการหนังสือตอบกลับได้อย่างรวดเร็วและเป็นระบบ
- **เป็นที่ปรึกษา:** ตอบข้อสงสัยเกี่ยวกับงานสารบรรณได้ทันทีด้วย AI Chatbot
""")

st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

st.subheader("วิธีการใช้งาน")
col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.markdown("### ✍️ ร่างหนังสือราชการ")
        st.markdown("เพียงพิมพ์สิ่งที่คุณต้องการจะสื่อสารในช่องข้อความ ระบบ AI จะทำการวิเคราะห์และร่างเนื้อหาฉบับสมบูรณ์ให้คุณทันที")
with col2:
    with st.container(border=True):
        st.markdown("### 📬 สร้างหนังสือตอบกลับ")
        st.markdown("อัปโหลดไฟล์ PDF ของหนังสือที่ได้รับ แล้ว AI จะช่วยวิเคราะห์และสร้างร่างหนังสือตอบกลับที่สอดคล้องกันให้โดยอัตโนมัติ")
with col3:
    with st.container(border=True):
        st.markdown("### 🤖 ที่ปรึกษาอัจฉริยะ")
        st.markdown("มีข้อสงสัยเกี่ยวกับรูปแบบ, ระเบียบ, หรือการใช้งาน ถาม AI Chatbot ของเราได้ตลอดเวลา")

st.markdown('</div>', unsafe_allow_html=True)