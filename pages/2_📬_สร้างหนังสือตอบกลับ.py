import streamlit as st
import ollama
import requests
import json
import io
import re
import numpy as np
import cv2
import streamlit.components.v1 as components

from skimage.filters import threshold_local
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from PIL import Image
from styles.main_style import load_css
from utils.ui_helper import render_sidebar 
from utils.llm_helper import (
    LLM_MODEL,
    replySec234_generation,
    extract_structured_data,
    init_ollama_client,
    post_process_ocr_text,
    replySec1_generation,
    get_extraction,
    create_docx_from_text,
    log_feedback_to_csv
)

# --- PAGE CONFIG & SETUP ---
st.set_page_config(
    page_title="สร้างหนังสือตอบกลับจากเอกสาร",
    page_icon="📬",
    layout="wide",
)

# สร้าง client และสถานะ
ollama_client, OLLAMA_AVAILABLE = init_ollama_client()

load_css()
render_sidebar()

# --- CONFIGURATIONS ---
TYPHOON_OCR_IMAGE_ENDPOINT = "http://typhoon-ocr:8000/process"

# --- GLOBAL VARIABLES & SESSION STATE INITIALIZATION ---
if 'ocr_text_content' not in st.session_state:
    st.session_state.ocr_text_content = None
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'current_doc_type_for_data' not in st.session_state:
    st.session_state.current_doc_type_for_data = None
if 'reply_content' not in st.session_state:
    st.session_state.reply_content = ""
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'confirmed_opening_paragraph' not in st.session_state:
    st.session_state.confirmed_opening_paragraph = ""
if 'full_reply_draft' not in st.session_state:
    st.session_state.full_reply_draft = ""
if 'is_draft_generated' not in st.session_state:
    st.session_state.is_draft_generated = False
if 'opening_options' not in st.session_state:
    st.session_state.opening_options = []
if 'opening_corrections_log' not in st.session_state:
    st.session_state.opening_corrections_log = []

FIELDS_MEMORANDUM = {
    "department": "ส่วนราชการ",
    "document_number": "ที่ (เลขหนังสือ)",
    "date": "วันที่",
    "subject": "เรื่อง",
    "recipient": "เรียน/เสนอ",
    "reference": "อ้างถึง",
    "attachments": "สิ่งที่ส่งมาด้วย",
    "body_main": "เนื้อความหลัก",
    "proposer_rank_name": "ยศ ชื่อผู้เสนอ",
    "proposer_position": "ตำแหน่งผู้เสนอ",
    "approver_rank_name": "ยศ ชื่อผู้อนุมัติ",
    "approver_position": "ตำแหน่งผู้อนุมัติ"
}

FIELDS_JOINT_NEWS_PAPER = {
    "urgency": "ลำดับความเร่งด่วน",
    "confidentiality": "ชั้นความลับ",
    "datetime_group": "หมู่-วัน-เวลา",
    "page_info": "หน้าที่",
    "originator_ref": "ที่ของผู้ให้ข่าว",
    "from_department": "จาก (หน่วยงานผู้ส่ง)",
    "to_recipient": "ถึง (ผู้รับปฏิบัติ)",
    "info_recipient": "ผู้รับทราบ",
    "body_main": "เนื้อหาข่าว",
    "qr_email": "QR Code/Email",
    "responsible_unit": "หน่วย (ผู้รับผิดชอบ)",
    "phone": "โทรศัพท์",
    # "reporter_signature": "ลายเซ็นผู้เขียนข่าว", # การสกัดลายเซ็นอาจยาก
    "reporter_rank_name_position": "ยศ ชื่อ ตำแหน่งผู้เขียนข่าว",
    # "approver_signature": "ลายเซ็นนายทหารอนุมัติข่าว", # การสกัดลายเซ็นอาจยาก
    "approver_rank_name_position": "ยศ ชื่อ ตำแหน่งนายทหารอนุมัติข่าว"
}

UNIT_ABBREVIATIONS = [
    "กธก.ศซบ.ทหาร", "กวก.ศซบ.ทหาร", "ผงป.นซบ.ทหาร", "ผกง.นซบ.ทหาร", "นตส.นซบ.ทหาร", "นธน.นซบ.ทหาร", "กกล.นซบ.ทหาร","กขซ.นซบ.ทหาร","กยก.นซบ.ทหาร", 
    "กตซ.นซบ.ทหาร", "สปก.นซบ.ทหาร", "กปก.๑ สปก.นซบ.ทหาร", "กปก.๒ สปก.นซบ.ทหาร", "กปก.๓ สปก.นซบ.ทหาร", "กปก.ศซบ.ทหาร", "ศซล.นซบ.ทหาร",
    "กวก.ศซล.นซบ.ทหาร", "รร.ซบ.ทหาร ศซล.นซบ.ทหาร", "กศษ.รร.ซบ.ทหาร ศซล.นซบ.ทหาร", "สน.บก.บก.ทท.", "สลก.บก.ทท.", "สจร.ทหาร", "สตน.ทหาร", "สสก.ทหาร",
    "สสก.บก.ทท.", "สยย.ทหาร", "ลชท.รอง", "ศปร.", "ศซบ.ทหาร", "สธน.ทหาร", "ขว.ทหาร", "ยก.ทหาร", "กบ.ทหาร", "กร.ทหาร", "กน.ทหาร", "สส.ทหาร", "กปท.ศทส.สส.ทหาร",
    "กทค.ศทท.สส.ทหาร", "พัน.ปสอ.สส.ทหาร", "ร้อย.บก.พัน.ส.บก.ทท.สส.ทหาร", "ร้อย.บก.พัน.ส.", "สปช.ทหาร", "นทพ.", "ศรภ.", "ศตก.", "สบ.ทหาร", "กง.ทหาร", "ผท.ทหาร",
    "ยบ.ทหาร", "สนพ.ยบ.ทหาร", "ชด.ทหาร", "บก.สปท.", "วปอ.สปท.", "วสท.สปท.", "สจว.สปท.", "ศศย.สปท.", "สศท.สปท.", "รร.ตท.สปท.", "รร.ชท.สปท.",
    "รอง ผอ.กภศ.ศศย.สปท.", "รอง ผอ.กกว.วสท.สปท.", "สน.พน.วสท.สปท.", "สน.พน.สปท.", "สน.รอง ผบ.สปท.", "สน.เสธ.สปท.", "กพ.ทหาร", "รอง จก.กพ.ทหาร",
    "จก.กพ.ทหาร", "บก.ทท.", "ผช.ผอ.กรภ.ศซบ.ทหาร", "หก.กธก.ศซบ.ทหาร", "กรภ.ศซบ.ทหาร", "กปก.ศซบ.ทหาร", "กวก.ศซบ.ทหาร", "นขต.ศซบ.ทหาร", "ผอ.ศซบ.ทหาร",
    "กยก.ศซบ.ทหาร", "ศซล.นซบ.ทหาร", "ผบ.นซบ.ทหาร", "ผอ.ศซล.นซบ.ทหาร", "ผอ.กรภ.ศซบ.ทหาร", "ผอ.กวก.ศซบ.ทหาร", "ศชป.ทหาร", "ผอ.กวก.ศซบ.ทหาร", "สสจ.ทหาร",
    "สนย.ทหาร", "วสส.สปท.", "สคท.สปท.", "กมศ.บก.สปท.", "รอง เสธ.สปท.", "ผบ.สปท.", "คทส.บก.ทหาร", "ผอ.กนผ.สผอ.สส.ทหาร", "ผอ.สผอ.สส.ทหาร", "จก.สส.ทหาร",
    "ผอ.กศช.สศท.สปท.", "เสธ.สปท.", "กวก.ศชล.นชบ.ทหาร", "สสท.ทร.", "ศชบ.สสท.ทร.", "จก.สสท.ทร.", "จก.สน.ทหาร", "สนพ.กพ.ทหาร","ศซล.นซบ.ทหาร"
]


# --- HELPER FUNCTIONS ---
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    try:
#         open_cv_image = np.array(pil_image.convert('RGB'))
#         open_cv_image = open_cv_image[:, :, ::-1].copy() # แปลงจาก RGB เป็น BGR

#         gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

#         coords = cv2.findNonZero(cv2.bitwise_not(gray_image))
#         angle = cv2.minAreaRect(coords)[-1]

#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle

#         # ทำการหมุนภาพเพื่อแก้ไขความเอียง
#         (h, w) = gray_image.shape[:2]
#         center = (w // 2, h // 2)
#         rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#         deskewed_image = cv2.warpAffine(gray_image, rotation_matrix, (w, h),
#                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#         denoised_image = cv2.medianBlur(deskewed_image, 3)
#         binary_image = cv2.adaptiveThreshold(denoised_image, 255,
#                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                              cv2.THRESH_BINARY, 31, 15)


#         final_pil_image = Image.fromarray(binary_image)
#         return final_pil_image
        return pil_image

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return pil_image

def ocr_from_images(image_bytes_list, file_name_for_log="image"):
    """Sends a list of image bytes to Typhoon-OCR and aggregates results."""
    full_text_from_all_images = [] 
    has_errors = False

    for i, img_bytes in enumerate(image_bytes_list):
        # st.info(f"OCR หน้าที่ {i+1}/{len(image_bytes_list)} ของไฟล์ '{file_name_for_log}'...")
        files = {'file': (f'page_{i+1}.png', img_bytes, 'image/png')}
        try:
            response = requests.post(TYPHOON_OCR_IMAGE_ENDPOINT, files=files, timeout=180) # Increased timeout
            response.raise_for_status()
            ocr_result_single_image = response.json()
            
            st.json(ocr_result_single_image) 

            current_image_text = ""
            if isinstance(ocr_result_single_image, dict) and "result" in ocr_result_single_image:
                current_image_text = ocr_result_single_image["result"]
                full_text_from_all_images.append(current_image_text.strip())
            else:
                st.warning(f"โครงสร้าง OCR output จากภาพหน้า {i+1} ไม่เป็นที่รู้จัก หรือไม่มี key 'result': {str(ocr_result_single_image)[:200]}...")
                pass


        except requests.exceptions.Timeout:
            st.error(f"หมดเวลาในการเชื่อมต่อ Typhoon-OCR สำหรับภาพหน้า {i+1}.")
            full_text_from_all_images.append(f"[OCR Timeout Page {i+1}]")
            has_errors = True
        except requests.exceptions.RequestException as e:
            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Typhoon-OCR สำหรับภาพหน้า {i+1}: {e}")
            full_text_from_all_images.append(f"[OCR Request Error Page {i+1}]")
            has_errors = True
        except json.JSONDecodeError:
            st.error(f"Typhoon-OCR ไม่ได้ตอบกลับเป็น JSON ที่ถูกต้องสำหรับภาพหน้า {i+1}. Response: {response.text[:200]}...")
            full_text_from_all_images.append(f"[OCR JSON Error Page {i+1}]")
            has_errors = True

    # รวม text จากทุกหน้าด้วยตัวคั่นที่ชัดเจน
    separator_template = "\n\n--- End of Page {page_num} / Total Pages {total_pages} ---\n\n"
    combined_text = ""
    total_pages_ocr = len(full_text_from_all_images)
    for i, page_text in enumerate(full_text_from_all_images):
        combined_text += page_text
        if i < total_pages_ocr - 1: # ไม่ต้องเพิ่ม separator หลังหน้าสุดท้าย
             combined_text += separator_template.format(page_num=i+1, total_pages=total_pages_ocr)

    return combined_text.strip(), has_errors

def on_radio_change():
    
    if 'opening_choice_radio' in st.session_state:
        st.session_state.selected_opening = st.session_state.opening_choice_radio
        st.session_state.reply_content = st.session_state.opening_choice_radio

def sync_opening_paragraph():
    
    selected_option = st.session_state.opening_choice_radio_selector
    st.session_state.reply_content = selected_option
    st.session_state.selected_opening = selected_option

    
    
    
# --- MAIN UI LAYOUT ---
with st.container(border=True):
    st.header("📬 สร้างหนังสือตอบกลับจากเอกสาร")
    st.write("อัปโหลดไฟล์ PDF ของ 'หนังสือรับ', ระบบจะทำ OCR, สกัดข้อมูล, และช่วยร่างหนังสือตอบกลับ")
    st.markdown("---")

    # --- Step 1: File Upload ---
    st.markdown("#### ขั้นตอนที่ 1: อัปโหลดไฟล์หนังสือรับ (PDF)")
    uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        # Check if it's a new file or the same one
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.ocr_text_content = None
            st.session_state.extracted_data = None
            st.session_state.reply_content = ""
            st.session_state.current_doc_type_for_data = None
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success(f"ไฟล์ใหม่: '{uploaded_file.name}'")


        # --- OCR Process ---
        if st.session_state.ocr_text_content is None:
            file_bytes = uploaded_file.getvalue()
            with st.status(f"⚙️ กำลังประมวลผล '{uploaded_file.name}'...", expanded=True) as status_ocr:
                try:
                    status_ocr.write("🔄 กำลังแปลง PDF เป็นรูปภาพ...")
                    pil_images = convert_from_bytes(file_bytes, dpi=300, fmt='png', thread_count=4)
                    image_bytes_list = []
                    for i, image in enumerate(pil_images):
                        image = preprocess_image(image)
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        image_bytes_list.append(img_byte_arr.getvalue())
                    status_ocr.update(label=f"✅ แปลง PDF เป็น {len(image_bytes_list)} รูปภาพสำเร็จ!", state="running")

                    status_ocr.write("🔍 กำลังทำ OCR จากรูปภาพ...")
                    ocr_text_from_func, ocr_errors = ocr_from_images(image_bytes_list, uploaded_file.name)

                    # --- เรียกใช้ Post-processing ---
                    if ocr_text_from_func: # ใช้ ocr_text_from_func ที่ได้จาก ocr_from_images
                        status_ocr.write("🔧 กำลังปรับปรุงผลลัพธ์ OCR...")
                        ocr_text_processed = post_process_ocr_text(ocr_text_from_func) 
                        st.session_state.ocr_text_content = ocr_text_processed 

                        # แสดงผลก่อนและหลังการ process เพื่อ debug
                        # with st.expander("แสดง OCR text ก่อนและหลัง process (Debug)"):
                        #     st.text_area("OCR Raw:", ocr_text_from_func, height=150)
                        #     st.text_area("OCR Processed:", ocr_text_processed, height=150)
                    else:
                        # ถ้า ocr_text_from_func เป็น None หรือสตริงว่าง ก็ให้ ocr_text_content เป็นค่าเดียวกัน
                        st.session_state.ocr_text_content = ocr_text_from_func

                    if ocr_errors:
                         status_ocr.update(label=f"⚠️ OCR สำเร็จบางส่วนสำหรับ '{uploaded_file.name}' (มีข้อผิดพลาดบางหน้า)", state="complete", expanded=False)
                    else:
                        status_ocr.update(label=f"✅ OCR ไฟล์ '{uploaded_file.name}' สำเร็จ!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในกระบวนการ OCR: {e}")
                    status_ocr.update(label=f"❌ เกิดข้อผิดพลาดในกระบวนการ OCR: {e}", state="error")
                    st.session_state.ocr_text_content = None
                    st.stop()

        # --- Display OCR Text and Select Document Type ---
        if st.session_state.ocr_text_content:
            with st.expander("แสดงตัวอย่างเนื้อหาจาก OCR (1000 ตัวอักษรแรก)"):
                st.text(st.session_state.ocr_text_content[:1000] + "..." if st.session_state.ocr_text_content and len(st.session_state.ocr_text_content) > 100 else st.session_state.ocr_text_content)

            st.markdown("#### ขั้นตอนที่ 1.1: โปรดระบุประเภทของหนังสือรับ")
            doc_types = ["บันทึกข้อความ", "กระดาษข่าวร่วม (ทท.)"]
            if st.session_state.current_doc_type_for_data and st.session_state.current_doc_type_for_data in doc_types:
                default_index_doc_type = doc_types.index(st.session_state.current_doc_type_for_data)
            else:
                default_index_doc_type = 0

            selected_doc_type = st.radio(
                "เลือกประเภทเอกสารที่อัปโหลด:",
                options=doc_types,
                index=default_index_doc_type,
                key="doc_type_selection_radio", # Unique key
                horizontal=True
            )

            # --- Data Extraction ---
            st.markdown("---")
            st.markdown("#### ขั้นตอนที่ 1.2: สกัดข้อมูลจากเอกสาร")

            # If document type changes, clear old extracted data
            if st.session_state.current_doc_type_for_data != selected_doc_type:
                st.session_state.extracted_data = None # Clear data if type changed
                st.info(f"ผู้ใช้เลือกประเภทเอกสารเป็น '{selected_doc_type}' หากเปลี่ยนกรุณากด 'สกัดข้อมูล' อีกครั้ง")


            col_extract1, col_extract2 = st.columns([3,1])
            with col_extract1:
                extract_button_label = "📊 สกัดข้อมูล"
                if st.session_state.extracted_data and st.session_state.current_doc_type_for_data == selected_doc_type:
                    extract_button_label = "🔄 สกัดข้อมูลอีกครั้ง"
                extract_button = st.button(extract_button_label, 
                                           use_container_width=True,
                                           disabled=not OLLAMA_AVAILABLE or not selected_doc_type or not st.session_state.ocr_text_content)
            with col_extract2:
                if st.session_state.extracted_data:
                    if st.button("🗑️ ล้างข้อมูลที่สกัด", use_container_width=True):
                        st.session_state.extracted_data = None
                        st.session_state.reply_content = ""


            if extract_button:
                st.session_state.current_doc_type_for_data = selected_doc_type
                st.session_state.extracted_data = None 
                st.session_state.reply_content = ""
                st.session_state.opening_options = [] 
                st.session_state.selected_opening = ""

                try:
                    with st.spinner(f"🧠 AI กำลังวิเคราะห์และสกัดข้อมูลสำหรับ '{selected_doc_type}'..."):

                        system_prompt, user_prompt_template, field_keys = get_extraction(selected_doc_type)

                        raw_extracted = extract_structured_data(
                            client=ollama_client,
                            ocr_text_content=st.session_state.ocr_text_content,
                            document_type=selected_doc_type,
                            system_prompt=system_prompt,
                            user_prompt_template=user_prompt_template
                        )

                        if raw_extracted and isinstance(raw_extracted, dict):
                            st.session_state.extracted_data = {
                                key: raw_extracted.get(key) for key in field_keys
                            }
                            st.success(f"สกัดข้อมูลสำหรับ '{selected_doc_type}' สำเร็จ!")
                        else:
                            st.session_state.extracted_data = None 
                            st.error("การสกัดข้อมูลคืนค่าที่ไม่คาดคิด (ไม่ใช่ Dictionary)")

                except Exception as e:
                    st.session_state.extracted_data = None 
                    st.error(f"AI ไม่สามารถสกัดข้อมูลได้: {e}")

                if not st.session_state.extracted_data and extract_button: 
                    st.warning("ไม่สามารถสกัดข้อมูลจากเอกสารได้ กรุณาตรวจสอบ Log หรือลองอีกครั้ง")


            # --- Display and Edit Extracted Data ---
            if st.session_state.extracted_data and st.session_state.current_doc_type_for_data == selected_doc_type:
                st.markdown("#### ขั้นตอนที่ 1.3: ตรวจสอบและแก้ไขข้อมูลที่สกัดได้")

                if selected_doc_type == "บันทึกข้อความ":
                    current_fields_map = FIELDS_MEMORANDUM
                elif selected_doc_type == "กระดาษข่าวร่วม (ทท.)":
                    current_fields_map = FIELDS_JOINT_NEWS_PAPER
                else:
                    current_fields_map = {}

                with st.form(key=f"edit_form_{selected_doc_type.replace(' ', '_').lower()}"):
                    st.subheader(f"ข้อมูลจาก: {selected_doc_type}")
                    form_cols = st.columns(2)
                    col_idx = 0
                    temp_edited_data = {}

                    for field_key, field_label in current_fields_map.items():
                        current_value = st.session_state.extracted_data.get(field_key)
                        display_value = str(current_value) if current_value is not None else ""

                        if field_key in ["body_main", "subject"]:
                            temp_edited_data[field_key] = st.text_area(
                                f"{field_label} ({field_key}):", value=display_value,
                                height=150 if field_key == "body_main" else 75,
                                key=f"edit_{selected_doc_type}_{field_key}"
                            )
                        else:
                            with form_cols[col_idx % 2]:
                                temp_edited_data[field_key] = st.text_input(
                                    f"{field_label} ({field_key}):", value=display_value,
                                    key=f"edit_{selected_doc_type}_{field_key}"
                                )
                            col_idx += 1

                    save_edits_button = st.form_submit_button("💾 บันทึกการแก้ไขข้อมูล", use_container_width=True) #here
#                     if save_edits_button:
#                         st.session_state.extracted_data.update(temp_edited_data)
#                         st.session_state.opening_options = []
# #                         st.session_state.reply_content = ""
#                         st.session_state.full_reply_draft = ""
#                         st.session_state.is_draft_generated = False
#                         st.success("บันทึกการแก้ไขข้อมูลเรียบร้อยแล้ว!")
#                         st.rerun() 
                    if save_edits_button:
                        # 1. เปรียบเทียบว่ามีการเปลี่ยนแปลงข้อมูลจริงหรือไม่
                        has_changed = any(str(st.session_state.extracted_data.get(k, '')) != str(v) for k, v in temp_edited_data.items())

                        # 2. อัปเดตข้อมูลที่สกัดได้
                        st.session_state.extracted_data.update(temp_edited_data)
                        st.success("บันทึกการแก้ไขข้อมูลเรียบร้อยแล้ว!")

                        # 3. ถ้ามีการเปลี่ยนแปลงและมี "ข้อ ๑" อยู่แล้ว ให้แจ้งเตือน
                        if has_changed and st.session_state.opening_options:
                            st.warning("⚠️ ข้อมูลหลักมีการเปลี่ยนแปลงแล้ว! เพื่อความถูกต้อง ควรสร้าง 'ข้อ ๑' ใหม่อีกครั้ง")


                # --- ขั้นตอนที่ 2: สร้างหนังสือตอบกลับ ---
                st.markdown("---")
                st.markdown("#### ขั้นตอนที่ 2: สร้างหนังสือตอบกลับ")

                # ส่วนที่ 2.1: สร้างตัวเลือก 'ข้อ ๑'
                st.markdown("##### ➡️ ขั้นตอนที่ 2.1: สร้างและยืนยัน 'ข้อ ๑'")

                generate_options_button = st.button(
                    "✨ สร้างตัวเลือก 'ข้อ ๑' ของหนังสือตอบกลับ",
                    use_container_width=True,
                    disabled=not OLLAMA_AVAILABLE or not st.session_state.extracted_data,
                    key=f"gen_opening_options_btn_{selected_doc_type}"
                )

                if generate_options_button:
                    st.session_state.reply_content = ""
                    st.session_state.opening_options = []
                    st.session_state.selected_opening = ""
                    st.session_state.full_reply_draft = ""
                    st.session_state.is_draft_generated = False

                    if st.session_state.extracted_data:
                        with st.spinner("AI กำลังสร้างตัวเลือกการเริ่มต้นหนังสือ (ข้อ ๑)..."):
                            st.session_state.opening_options = replySec1_generation(
                                                                    client=ollama_client,
                                                                    extracted_info=st.session_state.extracted_data, # ยังส่งไปเผื่อใช้ข้อมูลบางส่วน
                                                                    ocr_text_content=st.session_state.ocr_text_content, # ส่งเนื้อหา OCR ทั้งหมดไปด้วย
                                                                    num_options=3
                                                                )
                            st.session_state.last_extracted_for_opening = st.session_state.extracted_data.get('document_number')

                            if st.session_state.opening_options and all(isinstance(opt, str) for opt in st.session_state.opening_options):
                                st.session_state.selected_opening = st.session_state.opening_options[0]
                                st.session_state.reply_content = st.session_state.selected_opening
                                st.success("สร้างตัวเลือก 'ข้อ ๑' สำเร็จ!")
                            else:
                                st.error("AI ไม่สามารถสร้างตัวเลือก 'ข้อ ๑' ที่ถูกต้องได้ในขณะนี้ (รูปแบบไม่ถูกต้อง)")
                                st.session_state.opening_options = []
                    else:
                        st.error("ไม่พบข้อมูลที่สกัดได้ เพื่อใช้สร้างตัวเลือก 'ข้อ ๑'")


                # ส่วนที่ 2.2 และ 3 จะแสดงผลก็ต่อเมื่อมีตัวเลือก "ข้อ ๑" แล้วเท่านั้น
                if st.session_state.get('opening_options'):
                    st.markdown("###### ➡️ ขั้นตอนที่ 2.1.1: เลือกรูปแบบ 'ข้อ ๑' ที่ AI แนะนำ")
                    st.radio(
                        "เลือกรูปแบบ:",
                        options=st.session_state.opening_options,
                        index=st.session_state.opening_options.index(st.session_state.selected_opening) if st.session_state.selected_opening in st.session_state.opening_options else 0,
                        key="opening_choice_radio_selector", # <--- Key ใหม่สำหรับ widget ที่อยู่นอกฟอร์ม
                        on_change=sync_opening_paragraph,
                        label_visibility="collapsed" # ซ่อน Label "เลือกรูปแบบ:" เพราะมีหัวข้ออยู่แล้ว
                    )

                    # ส่วนที่ 2.1.2: แก้ไขและยืนยัน "ข้อ ๑" (อยู่ในฟอร์ม)
                    st.markdown("###### ➡️ ขั้นตอนที่ 2.1.2: แก้ไขและยืนยัน 'ข้อ ๑' ที่เลือก")
                    with st.form(key=f"edit_opening_form_{selected_doc_type}"):
                        
                        edited_opening_in_form = st.text_area(
                            "แก้ไขเนื้อหา (ถ้าต้องการ):",
                            value=st.session_state.reply_content, # ค่านี้จะถูกอัปเดตโดย on_change ของ radio
                            height=150,
                            key="opening_text_area_in_form",
                            label_visibility="collapsed"
                        )

                        save_edit_button = st.form_submit_button(
                            "💾 บันทึกและยืนยัน 'ข้อ ๑' นี้",
                            use_container_width=True
                        )

                        if save_edit_button:
                            confirmed_text = edited_opening_in_form
                            
                            if st.session_state.reply_content != confirmed_text:
                                from datetime import datetime
                                correction_log_entry = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "original_text": st.session_state.reply_content,
                                    "edited_text": confirmed_text,
                                    "document_type": selected_doc_type,
                                    "document_subject": st.session_state.extracted_data.get('subject', 'N/A')
                                }
                                log_feedback_to_csv(correction_log_entry)
                                st.session_state.opening_corrections_log.append(correction_log_entry)
                                st.toast("บันทึกข้อมูลการแก้ไขเรียบร้อยแล้ว!", icon="👍")
                            else:
                                st.toast("ยืนยัน 'ข้อ ๑' เรียบร้อย (ไม่มีการเปลี่ยนแปลง)", icon="✔️")

                            st.session_state.reply_content = confirmed_text
                            if confirmed_text in st.session_state.opening_options:
                                st.session_state.selected_opening = confirmed_text

                            st.success("บันทึกและยืนยัน 'ข้อ ๑' เรียบร้อยแล้ว")
                            st.rerun()

                    # --- แสดง Log การแก้ไข ---
                    if st.session_state.opening_corrections_log:
                        with st.expander("ดูประวัติการแก้ไข 'ข้อ ๑' (Feedback Log)"):
                            st.dataframe(st.session_state.opening_corrections_log)

                    st.markdown("---")

                    # ส่วนที่ 2.3: ให้ AI ร่างเนื้อหาส่วนที่เหลือ
                    st.markdown("##### ➡️ ขั้นตอนที่ 2.2: ให้ AI ช่วยร่างเนื้อหาส่วนที่เหลือ")
                    with st.container(border=True):
                        st.markdown("###### โปรดระบุข้อมูลสำหรับการร่างหนังสือตอบกลับ:")

                        # List ของหน่วยงาน
                        department_options = UNIT_ABBREVIATIONS

                        # List ของเจตนาการตอบกลับ
                        reply_intent_options = {
                            "อนุมัติ / เห็นชอบตามที่เสนอ": "อนุมัติ/เห็นชอบ",
                            "ปฏิเสธ / ไม่สามารถดำเนินการได้": "ปฏิเสธ/ไม่เห็นชอบ",
                            "ตอบรับทราบ (แจ้งว่าจะดำเนินการต่อ)": "ตอบรับทราบ",
                            "ส่งต่อเรื่อง / ประสานงานให้หน่วยงานอื่น": "ส่งต่อเรื่อง/ประสานงาน"
#                             "ขอข้อมูลเพิ่มเติม": "ขอข้อมูลเพิ่มเติม",
#                             "แจ้งผลการดำเนินการ": "แจ้งผลการดำเนินการ (ทั่วไป)"
                        }

                        col1, col2 = st.columns(2)
                        with col1:
                            selected_department = st.selectbox(
                                "เลือกหน่วยงานผู้ตอบ (หน่วยงานของเรา):",
                                options=department_options,
                                key="department_selector"
                            )

                        with col2:
                            selected_intent_display = st.selectbox(
                                "เลือกเจตนาหลักของการตอบกลับ:",
                                options=list(reply_intent_options.keys()), 
                                key="intent_selector"
                            )
                            selected_intent_for_llm = reply_intent_options[selected_intent_display]

                    # ปุ่มสำหรับเริ่มการร่างเนื้อหา
                    generate_full_draft_button = st.button(
                        "🤖 ให้ AI ช่วยร่างเนื้อหาต่อ (ข้อ ๒, ๓, ...)",
                        key=f"generate_full_draft_btn_{selected_doc_type}",
                        use_container_width=True,
                        disabled=not st.session_state.reply_content.strip() # ปิดปุ่มถ้า "ข้อ ๑" ว่าง
                    )

                    if generate_full_draft_button:

                        confirmed_text = st.session_state.get('reply_content', '')

                        st.session_state.confirmed_opening_paragraph = confirmed_text
                        st.session_state.is_draft_generated = False 

                        if not st.session_state.confirmed_opening_paragraph.strip():
                            st.error("เกิดข้อผิดพลาด: เนื้อหา 'ข้อ ๑' สำหรับการร่างต่อเป็นค่าว่าง กรุณากด 'บันทึกการแก้ไข' ก่อน")
                        else:
                            with st.spinner(f"AI กำลังร่างเนื้อหาต่อโดยมีเจตนาคือ '{selected_intent_display}'..."):
                                try:
                                    # 4. สร้าง dict สำหรับส่งให้ฟังก์ชัน
                                    info_for_analysis = st.session_state.extracted_data.copy()
                                    info_for_analysis["user_provided_opening_paragraph"] = st.session_state.confirmed_opening_paragraph

                                    our_department_name = selected_department
                                    reply_intent = selected_intent_for_llm

                                    generated_body = replySec234_generation(
                                        client=ollama_client,
                                        extracted_info=info_for_analysis,
                                        original_doc_type=selected_doc_type,
                                        reply_intent=reply_intent,
                                        relevant_internal_data={"our_department_name": our_department_name}
                                    )

                                    if generated_body and "เกิดข้อผิดพลาด" not in generated_body:
                                        st.session_state.full_reply_draft = generated_body
                                        st.session_state.is_draft_generated = True
                                        st.success("AI ร่างเนื้อหาทั้งหมดสำเร็จแล้ว! กรุณาตรวจสอบและแก้ไขด้านล่าง")
                                    else:
                                        st.error(f"AI ไม่สามารถสร้างเนื้อหาต่อได้ในขณะนี้: {generated_body}")
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาดระหว่างการร่างเนื้อหา: {e}")

                    # ส่วนที่ 3: แสดงผลฉบับร่างสมบูรณ์, แก้ไข, และ Export
                    if st.session_state.reply_content.strip() and st.session_state.get('is_draft_generated', False):
                        st.markdown("---")
                        st.markdown("#### ขั้นตอนที่ 3: ตรวจสอบและนำไปใช้งาน")
                        st.subheader("📝 ร่างหนังสือตอบกลับฉบับสมบูรณ์")

                        edited_full_draft = st.text_area(
                            "แก้ไขเนื้อหาฉบับร่าง:",
                            value=st.session_state.full_reply_draft,
                            height=400,
                            key=f"full_draft_textarea_{selected_doc_type}",
                            help="คุณสามารถแก้ไขเนื้อหาทั้งหมดได้ที่นี่ก่อนทำการ Export"
                        )
                        if edited_full_draft != st.session_state.full_reply_draft:
                            st.session_state.full_reply_draft = edited_full_draft

                        st.markdown("---")
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

    else: # No file uploaded
        if st.session_state.uploaded_file_name is not None:
            st.session_state.ocr_text_content = None
            st.session_state.extracted_data = None
            st.session_state.reply_content = ""
            st.session_state.current_doc_type_for_data = None
            st.session_state.uploaded_file_name = None
            st.session_state.opening_options = [] 
            st.session_state.selected_opening = ""
        st.info("กรุณาอัปโหลดไฟล์ PDF เพื่อเริ่มต้น")