import streamlit as st
import requests
import json
import io
import time
import re
import numpy as np
import cv2
import streamlit.components.v1 as components

from pdf2image import convert_from_bytes
from PIL import Image
from styles.main_style import load_css
from utils.ui_helper import render_sidebar, reset_workflow_states
from utils.llm_helper import (
    LLM_MODEL,
    replySec234_generation,
    extract_structured_data,
    init_ollama_client,
    post_process_ocr_text,
    replySec1_generation,
    get_extraction,
    create_docx_from_text,
    log_feedback_to_csv,
    FIELDS_MEMORANDUM,
    FIELDS_JOINT_NEWS_PAPER,
    CORRECT_UNIT_ABBREVIATIONS
)

# --- PAGE CONFIG & SETUP ---
st.set_page_config(
    page_title="สร้างหนังสือตอบกลับจากเอกสาร",
    page_icon="📬",
    layout="wide",
)

# --- INITIALIZATION ---
ollama_client, OLLAMA_AVAILABLE = init_ollama_client()
load_css()
render_sidebar()
TYPHOON_OCR_IMAGE_ENDPOINT = "http://typhoon-ocr:8000/process"

# --- SESSION STATE INITIALIZATION ---
states_to_init = {
    'ocr_text_content': None,
    'extracted_data': None,
    'current_doc_type_for_data': None,
    'reply_content': "",
    'uploaded_file_name': None,
    'confirmed_opening_paragraph': "",
    'full_reply_draft': "",
    'is_draft_generated': False,
    'opening_options': [],
    'selected_opening': "",
    'opening_corrections_log': []
}
for key, value in states_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- HELPER FUNCTIONS ---
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    try:
        open_cv_image = np.array(pil_image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy() # แปลงจาก RGB เป็น BGR

        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # --- Deskewing ---
        coords = cv2.findNonZero(cv2.bitwise_not(gray_image))
        if coords is None: # ถ้าเป็นภาพขาวล้วนหรือดำล้วน
            return pil_image 
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = gray_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed_image = cv2.warpAffine(gray_image, rotation_matrix, (w, h),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # --- Denoising and Binarization ---
        denoised_image = cv2.medianBlur(deskewed_image, 3)
        binary_image = cv2.adaptiveThreshold(denoised_image, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 15)

        final_pil_image = Image.fromarray(binary_image)
#         return final_pil_image

        # --- testing ---
        return pil_image 

    except Exception as e:
        st.warning(f"เกิดข้อผิดพลาดระหว่าง Image Preprocessing: {e}. ใช้ภาพต้นฉบับแทน")
        return pil_image

def ocr_from_images(image_bytes_list, file_name_for_log="image"):
    """Sends a list of image bytes to Typhoon-OCR and aggregates results."""
    full_text_from_all_images = []
    has_errors = False
    progress_bar = st.progress(0, text="กำลังทำ OCR...")

    for i, img_bytes in enumerate(image_bytes_list):
        files = {'file': (f'page_{i+1}.png', img_bytes, 'image/png')}
        try:
            response = requests.post(TYPHOON_OCR_IMAGE_ENDPOINT, files=files, timeout=180)
            response.raise_for_status()
            ocr_result_single_image = response.json()
            if isinstance(ocr_result_single_image, dict) and "result" in ocr_result_single_image:
                full_text_from_all_images.append(ocr_result_single_image["result"].strip())
        except Exception as e:
            st.warning(f"เกิดข้อผิดพลาดในการ OCR หน้า {i+1}: {str(e)[:100]}...")
            has_errors = True
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(image_bytes_list), text=f"กำลังทำ OCR หน้าที่ {i+1}/{len(image_bytes_list)}")
    
    progress_bar.empty()
    separator = "\n\n--- End of Page ---\n\n"
    return separator.join(full_text_from_all_images), has_errors

def sync_opening_paragraph():
    """Syncs the radio button choice to the text area."""
    selected_option = st.session_state.get("opening_choice_radio_selector")
    if selected_option:
        st.session_state.reply_content = selected_option
        st.session_state.selected_opening = selected_option

# --- MAIN UI LAYOUT ---
with st.container(border=True):
    st.header("📬 สร้างหนังสือตอบกลับจากเอกสาร")
    st.write("อัปโหลดไฟล์ PDF ของ 'หนังสือรับ', ระบบจะทำ OCR, สกัดข้อมูล, และช่วยร่างหนังสือตอบกลับ")

    with st.expander("⚙️ การตั้งค่าขั้นสูง (Advanced Settings)"):
        use_fuzzy_matching = st.checkbox("เปิดใช้งานการแก้ไขคำผิดขั้นสูง (Fuzzy Matching)", value=False, help="หากเปิดใช้งาน ระบบจะพยายามแก้ไขชื่อย่อหน่วยงานที่ OCR ผิดเพี้ยนเล็กน้อยให้ถูกต้อง อาจทำให้การประมวลผล OCR ช้าลงเล็กน้อย")
    st.markdown("---")

    st.markdown("#### ขั้นตอนที่ 1: อัปโหลดไฟล์หนังสือรับ (PDF)")
    uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        
        is_new_file = st.session_state.uploaded_file_name != uploaded_file.name

        if is_new_file:
            reset_workflow_states()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success(f"ไฟล์ใหม่: '{uploaded_file.name}'")
            st.rerun()

        # --- OCR Process ---
        if not st.session_state.ocr_text_content:
            with st.spinner(f"กำลังประมวลผลไฟล์ '{uploaded_file.name}'..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    pil_images = convert_from_bytes(file_bytes, dpi=300, fmt='png', thread_count=4)
                    
                    image_bytes_list = []
                    for image in pil_images:
                        processed_img = preprocess_image(image)
                        img_byte_arr = io.BytesIO()
                        processed_img.save(img_byte_arr, format='PNG')
                        image_bytes_list.append(img_byte_arr.getvalue())

                    ocr_text_from_func, ocr_errors = ocr_from_images(image_bytes_list, uploaded_file.name)
                    ocr_text_processed = post_process_ocr_text(ocr_text_from_func, fuzzy_enabled=use_fuzzy_matching)
                    st.session_state.ocr_text_content = ocr_text_processed
                    st.rerun()
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดร้ายแรงในกระบวนการ OCR: {e}")
                    st.session_state.ocr_text_content = None # Ensure state is cleared on error

        # --- Main Workflow (executes only if OCR content exists) ---
        if st.session_state.ocr_text_content:
            with st.expander("แสดงตัวอย่างเนื้อหาจาก OCR", expanded=True):
                st.text_area("OCR Content:", st.session_state.ocr_text_content, height=200, disabled=True, label_visibility="collapsed")

            st.markdown("#### ขั้นตอนที่ 1.1: โปรดระบุประเภทของหนังสือรับ")
            doc_types = ["บันทึกข้อความ", "กระดาษข่าวร่วม (ทท.)"]
            
            # Set default index for radio button
            try:
                default_index = doc_types.index(st.session_state.current_doc_type_for_data)
            except (ValueError, TypeError):
                default_index = 0

            selected_doc_type = st.radio("เลือกประเภทเอกสารที่อัปโหลด:", options=doc_types, index=default_index, horizontal=True)

            # --- Data Extraction ---
            st.markdown("---")
            st.markdown("#### ขั้นตอนที่ 1.2: สกัดข้อมูลจากเอกสาร")

            if st.session_state.current_doc_type_for_data != selected_doc_type:
                st.session_state.extracted_data = None
                st.info(f"ผู้ใช้เลือกประเภทเอกสารเป็น '{selected_doc_type}'. หากถูกต้องแล้ว กรุณากด 'สกัดข้อมูล'")

            col_extract1, col_extract2 = st.columns([3, 1])
            with col_extract1:
                extract_button_label = "🔄 สกัดข้อมูลอีกครั้ง" if st.session_state.extracted_data else "📊 สกัดข้อมูล"
                extract_button = st.button(extract_button_label, use_container_width=True, disabled=not OLLAMA_AVAILABLE)
            
            with col_extract2:
                if st.session_state.extracted_data:
                    if st.button("🗑️ ล้างข้อมูล", use_container_width=True):
                        st.session_state.extracted_data = None
                        st.rerun()

            if extract_button:
                st.session_state.current_doc_type_for_data = selected_doc_type
                with st.spinner(f"🧠 AI กำลังวิเคราะห์และสกัดข้อมูลสำหรับ '{selected_doc_type}'..."):
                    try:
                        system_prompt, user_prompt_template, field_keys = get_extraction(selected_doc_type)
                        raw_extracted = extract_structured_data(ollama_client, st.session_state.ocr_text_content, selected_doc_type, system_prompt, user_prompt_template)
                        
                        if raw_extracted and isinstance(raw_extracted, dict):
                             st.session_state.extracted_data = {key: raw_extracted.get(key) for key in field_keys}
                             st.success("สกัดข้อมูลสำเร็จ!")
                             time.sleep(1) # Short pause to let user see the success message
                             st.rerun()
                        else:
                            st.error("AI ไม่สามารถสกัดข้อมูลในรูปแบบที่ถูกต้องได้")
                            st.session_state.extracted_data = None

                    except Exception as e:
                        st.error(f"AI ไม่สามารถสกัดข้อมูลได้: {e}")
                        st.session_state.extracted_data = None

            # --- Display and Edit Extracted Data (and subsequent steps) ---
            if st.session_state.extracted_data:
                st.markdown("#### ขั้นตอนที่ 1.3: ตรวจสอบและแก้ไขข้อมูลที่สกัดได้")
                current_fields_map = FIELDS_MEMORANDUM if selected_doc_type == "บันทึกข้อความ" else FIELDS_JOINT_NEWS_PAPER

                with st.form(key=f"edit_form_{selected_doc_type.replace(' ', '_')}"):
                    st.subheader(f"ข้อมูลจาก: {selected_doc_type}")
                    form_cols = st.columns(2)
                    col_idx = 0
                    temp_edited_data = {}

                    for field_key, field_label in current_fields_map.items():
                        current_value = st.session_state.extracted_data.get(field_key, "")
                        display_value = str(current_value) if current_value is not None else ""
                        
                        if field_key in ["body_main", "subject"]:
                             temp_edited_data[field_key] = st.text_area(f"{field_label} ({field_key}):", value=display_value, height=150)
                        else:
                            with form_cols[col_idx % 2]:
                                temp_edited_data[field_key] = st.text_input(f"{field_label} ({field_key}):", value=display_value)
                            col_idx += 1
                    
                    if st.form_submit_button("💾 บันทึกการแก้ไขข้อมูล", use_container_width=True):
                        has_changed = any(str(st.session_state.extracted_data.get(k, '')) != str(v) for k, v in temp_edited_data.items())
                        st.session_state.extracted_data.update(temp_edited_data)
                        st.success("บันทึกการแก้ไขเรียบร้อยแล้ว!")
                        if has_changed and st.session_state.opening_options:
                            st.warning("ข้อมูลหลักมีการเปลี่ยนแปลง! เพื่อความถูกต้อง ควรสร้าง 'ข้อ ๑' ใหม่อีกครั้ง")
                        time.sleep(1.5)
                        st.rerun()

                # --- Step 2: Generate Reply ---
                st.markdown("---")
                st.markdown("#### ขั้นตอนที่ 2: สร้างหนังสือตอบกลับ")
                st.markdown("##### ➡️ ขั้นตอนที่ 2.1: สร้างและยืนยัน 'ข้อ ๑'")
                
                if st.button("✨ สร้างตัวเลือก 'ข้อ ๑' ของหนังสือตอบกลับ", use_container_width=True):
                    with st.spinner("AI กำลังสร้างตัวเลือกการเริ่มต้นหนังสือ (ข้อ ๑)..."):
                        options = replySec1_generation(ollama_client, st.session_state.extracted_data, st.session_state.ocr_text_content)
                        if options:
                            st.session_state.opening_options = options
                            st.session_state.selected_opening = options[0]
                            st.session_state.reply_content = options[0]
                            st.success("สร้างตัวเลือก 'ข้อ ๑' สำเร็จ!")
                            st.rerun()
                        else:
                            st.error("AI ไม่สามารถสร้างตัวเลือก 'ข้อ ๑' ที่ถูกต้องได้")

                if st.session_state.opening_options:
                    st.markdown("###### ➡️ 2.1.1: เลือกรูปแบบ 'ข้อ ๑' ที่ AI แนะนำ")
                    st.radio(
                        "เลือกรูปแบบ:",
                        options=st.session_state.opening_options,
                        index=st.session_state.opening_options.index(st.session_state.selected_opening) if st.session_state.selected_opening in st.session_state.opening_options else 0,
                        key="opening_choice_radio_selector",
                        on_change=sync_opening_paragraph,
                        label_visibility="collapsed"
                    )

                    st.markdown("###### ➡️ 2.1.2: แก้ไขและยืนยัน 'ข้อ ๑' ที่เลือก")
                    with st.form(key=f"edit_opening_form_{selected_doc_type}"):
                        edited_opening = st.text_area("แก้ไขเนื้อหา (ถ้าต้องการ):", value=st.session_state.reply_content, height=150, label_visibility="collapsed")
                        
                        if st.form_submit_button("💾 บันทึกและยืนยัน 'ข้อ ๑' นี้", use_container_width=True):
                            if st.session_state.reply_content != edited_opening:
                                log_entry = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "original_text": st.session_state.reply_content,
                                    "edited_text": edited_opening,
                                    "document_type": selected_doc_type,
                                    "document_subject": st.session_state.extracted_data.get('subject', 'N/A')
                                }
                                log_feedback_to_csv(log_entry)
                                st.session_state.opening_corrections_log.append(log_entry)
                                st.toast("บันทึกข้อมูลการแก้ไขเรียบร้อยแล้ว!", icon="👍")
                            
                            st.session_state.confirmed_opening_paragraph = edited_opening
                            st.success("ยืนยัน 'ข้อ ๑' เรียบร้อยแล้ว")
                            st.rerun()

                    if st.session_state.opening_corrections_log:
                        with st.expander("ดูประวัติการแก้ไข 'ข้อ ๑' (Feedback Log)"):
                            st.dataframe(st.session_state.opening_corrections_log)

                if st.session_state.confirmed_opening_paragraph:
                    st.markdown("---")
                    st.markdown("##### ➡️ ขั้นตอนที่ 2.2: ให้ AI ช่วยร่างเนื้อหาส่วนที่เหลือ")
                    with st.container(border=True):
                        st.markdown("###### โปรดระบุข้อมูลสำหรับการร่างหนังสือตอบกลับ:")
                        department_options = CORRECT_UNIT_ABBREVIATIONS
                        reply_intent_options = {
                            "อนุมัติ / เห็นชอบตามที่เสนอ": "อนุมัติ/เห็นชอบ",
                            "ปฏิเสธ / ไม่สามารถดำเนินการได้": "ปฏิเสธ/ไม่เห็นชอบ",
                            "ตอบรับทราบ (แจ้งว่าจะดำเนินการต่อ)": "ตอบรับทราบ",
                            "ส่งต่อเรื่อง / ประสานงานให้หน่วยงานอื่น": "ส่งต่อเรื่อง/ประสานงาน"
                        }

                        col1, col2 = st.columns(2)
                        with col1:
                            selected_department = st.selectbox("เลือกหน่วยงานผู้ตอบ:", options=department_options)
                        with col2:
                            selected_intent_display = st.selectbox("เลือกเจตนาหลัก:", options=list(reply_intent_options.keys()))
                            selected_intent_for_llm = reply_intent_options[selected_intent_display]
                    
                    if st.button("🤖 ให้ AI ช่วยร่างเนื้อหาต่อ (ข้อ ๒, ๓, ...)", use_container_width=True):
                        with st.spinner(f"AI กำลังร่างเนื้อหาต่อโดยมีเจตนาคือ '{selected_intent_display}'..."):
                            try:
                                info_for_analysis = st.session_state.extracted_data.copy()
                                info_for_analysis["user_provided_opening_paragraph"] = st.session_state.confirmed_opening_paragraph
                                
                                generated_body = replySec234_generation(
                                    client=ollama_client,
                                    extracted_info=info_for_analysis,
                                    original_doc_type=selected_doc_type,
                                    reply_intent=selected_intent_for_llm,
                                    relevant_internal_data={"our_department_name": selected_department}
                                )
                                st.session_state.full_reply_draft = st.session_state.confirmed_opening_paragraph + "\n" + generated_body
                                st.session_state.is_draft_generated = True
                                st.success("AI ร่างเนื้อหาทั้งหมดสำเร็จแล้ว!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"เกิดข้อผิดพลาดระหว่างการร่างเนื้อหา: {e}")

                if st.session_state.is_draft_generated:
                    st.markdown("---")
                    st.markdown("#### ขั้นตอนที่ 3: ตรวจสอบและนำไปใช้งาน")
                    st.text_area("📝 ร่างหนังสือตอบกลับฉบับสมบูรณ์:", value=st.session_state.full_reply_draft, height=400, key="full_draft_textarea")
                    
                    export_col1, export_col2 = st.columns(2)
                    with export_col1:
                            try:
                                subject = st.session_state.extracted_data.get('subject', 'เรื่องทั่วไป').replace("/", "_")
                                file_name_suggestion = f"ร่างตอบกลับ_{subject[:25]}.docx"

                                docx_bytes = create_docx_from_text(st.session_state.full_reply_draft)
                                st.download_button(
                                    label="📄 ดาวน์โหลดเป็น .docx",
                                    data=docx_bytes,
                                    file_name=file_name_suggestion,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True,
                                    disabled=not st.session_state.full_reply_draft.strip()
                                )
                            except Exception as e:
                                st.error(f"ไม่สามารถสร้างไฟล์ .docx ได้: {e}")

                    with export_col2:

                        text_to_copy = st.session_state.full_reply_draft

                        if text_to_copy and text_to_copy.strip():

                            js_string = text_to_copy.replace('\\', '\\\\').replace('`', '\\`').replace('\n', '\\n').replace('"', '\\"')

                            import streamlit.components.v1 as components
                            components.html(
                                f"""
                                <script>
                                // ฟังก์ชันนี้จะถูกเรียกเมื่อมีการคลิกปุ่ม HTML
                                function copyToClipboard() {{
                                    const text = `{js_string}`;
                                    // ใช้ API ของ Clipboard
                                    navigator.clipboard.writeText(text).then(() => {{
                                        // ส่ง custom event กลับไปหา Streamlit เพื่อแสดง Toast
                                        window.parent.parent.postMessage({{
                                            'type': 'streamlit:setFrameHeight',
                                            'data': {{'message': 'คัดลอกเนื้อหาเรียบร้อยแล้ว!', 'icon': '📋', 'type': 'toast'}}
                                        }}, '*');
                                    }}, (err) => {{
                                        alert("การคัดลอกล้มเหลว: " + err);
                                    }});
                                }}
                                </script>

                                <!-- สร้างปุ่ม HTML ที่เรียกใช้ฟังก์ชัน JS ข้างต้น -->
                                <button onclick="copyToClipboard()" 
                                        style="
                                            width: 100%; 
                                            padding: 0.60rem 0.75rem; /* ทำให้ขนาดใกล้เคียงกับปุ่ม Streamlit */
                                            border-radius: 0.5rem; 
                                            border: 1px solid rgba(49, 51, 63, 0.2);
                                            background-color: #FFFFFF; 
                                            color: #31333F; 
                                            font-weight: 400; 
                                            cursor: pointer; 
                                            transition: all 0.2s;
                                            font-family: 'Source Sans Pro', sans-serif; /* ใช้ฟอนต์เหมือน Streamlit */
                                            font-size: 1rem;
                                            text-align: center;
                                            line-height: 1.6;
                                        ">
                                    📋 คัดลอกเนื้อหาทั้งหมด
                                </button>
                                <style>
                                    /* เพิ่ม hover effect ให้ปุ่ม */
                                    button:hover {{
                                        border-color: #FF4B4B;
                                        color: #FF4B4B;
                                    }}
                                    button:active {{
                                        background-color: #F0F2F6;
                                    }}
                                </style>
                                """,
                                height=55
                            )
                        else:
                            st.button("📋 คัดลอกเนื้อหาทั้งหมด", use_container_width=True, disabled=True)

    else:
        # This block runs when no file is uploaded.
        if st.session_state.uploaded_file_name:
            reset_workflow_states() # Clear state if file is removed
        st.info("กรุณาอัปโหลดไฟล์ PDF เพื่อเริ่มต้น")