import streamlit as st # <<<--- เพิ่มบรรทัดนี้
import base64
import pandas as pd
import datetime
from pathlib import Path

# --- HELPER & CORE FUNCTIONS ---
@st.cache_data
def image_to_base64(image_path: Path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return None

# OLLAMA CONNECTION
OLLAMA_HOST = 'http://ollama:11434'
LLM_MODEL = 'scb10x/llama3.1-typhoon2-8b-instruct:latest'
try:
    client = ollama.Client(host=OLLAMA_HOST)
    client.list()
    OLLAMA_AVAILABLE = True
except Exception as e:
    OLLAMA_AVAILABLE = False
    print(f"Ollama connection error: {e}")


def save_feedback(original_text, edited_text, user_prompt):
    feedback_file = Path("feedback_log.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_feedback = pd.DataFrame({
        "timestamp": [timestamp],
        "user_prompt": [user_prompt],
        "original_ai_text": [original_text],
        "edited_user_text": [edited_text]
    })
    
    try:
        if feedback_file.exists():
            feedback_df = pd.read_csv(feedback_file)
            feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        else:
            feedback_df = new_feedback
        
        feedback_df.to_csv(feedback_file, index=False, encoding='utf-8-sig') # เพิ่ม encoding สำหรับภาษาไทย
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False

def text_extraction(file_storage):
    """ฟังก์ชันสำหรับสกัดข้อความจากไฟล์ PDF ที่อัปโหลด"""
    try:
        pdf_document = fitz.open(stream=file_storage.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ PDF ได้: {e}")
        return None

def image_to_base64(image_path: Path):
    """โหลดไฟล์ภาพและแปลงเป็น Base64 string"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        # คุณสามารถใส่โลโก้สำรองแบบ Base64 ที่นี่ได้ หรือแค่คืนค่า None
        return None