import streamlit as st
import ollama

from styles.main_style import load_css
from utils.ui_helper import render_sidebar
from utils.llm_helper import init_ollama_client, draft_generation

ollama_client, OLLAMA_AVAILABLE = init_ollama_client()

# --- PAGE CONFIG & SETUP ---
st.set_page_config(
    page_title="ร่างหนังสือราชการ",
    page_icon="✍️",
    layout="wide",
)

load_css()
render_sidebar()

with st.container(border=True):
    st.header("✍️ ร่างหนังสือราชการ")
    st.write("เพียงป้อนสิ่งที่คุณต้องการสื่อสารด้วยภาษาปกติ แล้ว AI จะแปลงเป็นเนื้อหาสำหรับหนังสือราชการให้คุณ")
    st.markdown("---")

# --- MAIN CONTENT ---
st.markdown("---")

col1, col2 = st.columns([3, 1.5])

with col1:
    with st.container(border=True):
        st.subheader("ขั้นตอนที่ 1: ใส่เนื้อหาที่ต้องการ")
        user_prompt = st.text_area("เนื้อหา", placeholder="ตัวอย่าง: แจ้งผลการอบรม...", height=280, label_visibility="collapsed")

with col2:
    with st.container(border=True):
        st.subheader("ขั้นตอนที่ 2: ตั้งค่า")
        
        doc_type = st.selectbox(
            "เลือกประเภทเอกสาร",
            ("กระดาษข่าวร่วม (ทท.)", "บันทึกข้อความ"),
            key="doc_type_selector"
        )
        
        # --- เพิ่มส่วนนี้: แสดงตัวเลือกนี้เมื่อเป็น "บันทึกข้อความ" เท่านั้น ---
        doc_salutation = ""
        if doc_type == "บันทึกข้อความ":
            doc_salutation = st.radio(
                "คำขึ้นต้น",
                ("เรียน", "เสนอ"),
                horizontal=True
            )
        # --- จบส่วนที่เพิ่ม ---

        formality_level = st.selectbox(
            "ระดับความเป็นทางการ", 
            ("เป็นทางการ", "เป็นทางการมากที่สุด")
        )


st.write("")
generate_button = st.button("✨ แปลงเป็นภาษาราชการ", use_container_width=True, disabled=not OLLAMA_AVAILABLE)

# --- Initialize Session State for the Workflow ---
if 'editing_mode' not in st.session_state:
    st.session_state.editing_mode = False
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = ""
if 'original_ai_text' not in st.session_state:
    st.session_state.original_ai_text = ""
if 'user_prompt_for_feedback' not in st.session_state:
    st.session_state.user_prompt_for_feedback = ""

# --- Logic for Generating Content ---
if generate_button and user_prompt:
    with st.spinner("🧠 AI กำลังประมวลผลและร่างเนื้อหา..."):
        generated_text = draft_generation(ollama_client, user_prompt, doc_type, formality_level, doc_salutation)
        st.session_state.generated_content = generated_text
        st.session_state.original_ai_text = generated_text # Store the pristine AI output
        st.session_state.user_prompt_for_feedback = user_prompt
        st.session_state.editing_mode = False # Always reset to read-only mode after new generation

# --- Display Results and Actions ---
if st.session_state.generated_content:
    st.write("")
    with st.container(border=True):
        st.subheader("ผลลัพธ์จาก AI")

        textarea_class = "editing" if st.session_state.editing_mode else "readonly"

        st.markdown(f'<div class="stTextArea {textarea_class}">', unsafe_allow_html=True)
        edited_text = st.text_area(
            "เนื้อหา:",
            value=st.session_state.generated_content,
            height=300,
            key="editable_content",
            disabled=not st.session_state.editing_mode
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")

        action_col1, action_col2 = st.columns(2)

        if st.session_state.editing_mode:
            # --- Editing State UI ---
            with action_col1:
                if st.button("✔️ บันทึกและส่ง Feedback", use_container_width=True, type="primary"):
                    if edited_text != st.session_state.original_ai_text:
                        if save_feedback(st.session_state.original_ai_text, edited_text, st.session_state.user_prompt_for_feedback):
                            st.toast("✔️ ขอบคุณสำหรับ Feedback!", icon="🎉")
                            st.session_state.generated_content = edited_text
                        else:
                            st.toast("❌ เกิดข้อผิดพลาดในการบันทึก Feedback", icon="🔥")
                    else:
                        st.toast("ℹ️ ไม่มีการเปลี่ยนแปลงเนื้อหา", icon="💡")

                    st.session_state.editing_mode = False
                    st.rerun()

            with action_col2:
                if st.button("ยกเลิก", use_container_width=True):
                    st.session_state.editing_mode = False
                    # Restore the original text if user cancels
                    st.session_state.generated_content = st.session_state.original_ai_text
                    st.rerun()

        else:
            # --- Default (Read-only) State UI ---
            with action_col1:
                if st.button("✏️ แก้ไขเนื้อหา", use_container_width=True):
                    st.session_state.editing_mode = True
                    st.rerun()

            with action_col2:
                # Prepare the string for JavaScript, escaping necessary characters
                js_string = edited_text.replace('\\', '\\\\').replace('`', '\\`').replace('\n', '\\n')

                # Create the copy button component with corrected f-string
                st.components.v1.html(
                    f"""
                    <script>
                    function copyToClipboard() {{
                        const text = `{js_string}`;
                        navigator.clipboard.writeText(text).then(() => {{
                            window.parent.postMessage({{
                                'type': 'streamlit:toast',
                                'data': {{'message': 'คัดลอกเนื้อหาเรียบร้อยแล้ว!', 'icon': '📋'}}
                            }}, '*');
                        }});
                    }}
                    </script>
                    <button onclick="copyToClipboard()" style="width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #B0BEC5; background-color: #F8F9FA; color: #3D405B; font-weight: 500; cursor: pointer; transition: all 0.2s;">
                        📋 คัดลอกเนื้อหา
                    </button>
                    """,
                    height=50,
                )