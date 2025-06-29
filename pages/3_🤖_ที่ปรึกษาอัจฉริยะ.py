import streamlit as st
from pathlib import Path
import time
import ollama

from styles.main_style import load_css 
from utils.ui_helper import render_sidebar
from utils.file_helper import image_to_base64
from utils.llm_helper import init_ollama_client, call_chatbot

ollama_client, OLLAMA_AVAILABLE = init_ollama_client()

# --- PAGE CONFIG & SETUP ---
st.set_page_config(
    page_title="ที่ปรึกษาอัจฉริยะ",
    page_icon="🤖",
    layout="wide",
)

load_css()
render_sidebar()

with st.container(border=True):
    st.header("🤖 ที่ปรึกษาอัจฉริยะ")
    st.write("สอบถามข้อสงสัยเกี่ยวกับการใช้งานระบบ, รูปแบบเอกสาร, หรือระเบียบที่เกี่ยวข้องได้ที่นี่")
    st.markdown("---")

    
def handle_new_question(prompt_text):
    if prompt_text:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        st.rerun()    
        
# main function

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ มีอะไรให้ผมช่วยเหลือ? สามารถเลือกจากคำถามที่พบบ่อย หรือพิมพ์คำถามของคุณได้เลยครับ"}]
    
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
    if message["role"] == "assistant" and i > 0:
        st.markdown("---")

if len(st.session_state.messages) <= 1:
    st.markdown("---")
    st.markdown("##### หรือเลือกจากคำถามที่พบบ่อย:")
    
    faq_col1, faq_col2 = st.columns(2)
    faq_questions = {
        "การใช้งานระบบ": [
            "จะทำหนังสือตอบกลับในระบบนี้ต้องทำยังไงบ้าง?",
            "ถ้าจะแก้ข้อความที่ AI สร้างให้ในเมนูร่างหนังสือราชการ ต้องกดปุ่มไหน?"
        ],
        "ระเบียบงานสารบรรณ": [
            "อ้างถึงงานระเบียบสารบรรณ การใช้คำขึ้นต้น 'เรียน' กับ 'เสนอ' ต่างกันอย่างไร?",
            "อ้างถึงงานระเบียบสารบรรณ แบบหนังสือภายในใช้กระดาษบันทึกข้อความ คืออะไร?"
        ]
    }

    with faq_col1:
        st.markdown(f"###### ⚙️ {list(faq_questions.keys())[0]}")
        for q in faq_questions["การใช้งานระบบ"]:
            if st.button(q, use_container_width=True):
                handle_new_question(q)

    with faq_col2:
        st.markdown(f"###### 📖 {list(faq_questions.keys())[1]}")
        for q in faq_questions["ระเบียบงานสารบรรณ"]:
            if st.button(q, use_container_width=True):
                handle_new_question(q)

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("🧠 กำลังประมวลผลหาคำตอบ..."):
            response = call_chatbot(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

            
if prompt := st.chat_input("พิมพ์คำถามของคุณ...", disabled=not OLLAMA_AVAILABLE):
    handle_new_question(prompt)

