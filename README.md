# ระบบสร้างเอกสารราชการอัตโนมัติ (Automatic Official Document Generation System)

![Rtarf Ai](https://img.shields.io/badge/RTARF-AI%20Innovator-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)
![Ollama](https://img.shields.io/badge/Ollama-Enabled-lightgrey.svg)

**ระบบสร้างเอกสารราชการอัตโนมัติ** คือเว็บแอปพลิเคชันที่ถูกสร้างขึ้นเพื่อปฏิวัติกระบวนการทำงานเอกสารราชการไทย โดยใช้เทคโนโลยีปัญญาประดิษฐ์ (AI) และ Large Language Models (LLM) เป็นแกนหลัก มีเป้าหมายเพื่อลดภาระงานด้านเอกสารที่ซ้ำซ้อน, เพิ่มความเร็วและความแม่นยำ, และทำหน้าที่เป็นผู้ช่วยอัจฉริยะสำหรับเจ้าหน้าที่

---

## ✨ คุณสมบัติหลัก (Core Features)

ระบบประกอบด้วย 3 โมดูลหลักที่ทำงานร่วมกันอย่างเป็นระบบ:

### ✍️ 1. ร่างหนังสือราชการ (Draft Generation)
- **หน้าที่:** แปลงข้อความภาษาพูด หรือ "ความคิด" ของผู้ใช้ ให้กลายเป็นเนื้อหาของหนังสือราชการ (`บันทึกข้อความ`, `กระดาษข่าวร่วม (ทท.)`) ที่มีโครงสร้างและสำนวนถูกต้องตามระเบียบ
- **เทคโนโลยี:** Prompt Engineering, LLM Generation

### 📬 2. สร้างหนังสือตอบกลับ (Reply Generation Pipeline)
- **หน้าที่:** กระบวนการอัตโนมัติแบบครบวงจร (End-to-End) ที่วิเคราะห์หนังสือรับในรูปแบบไฟล์ PDF และสร้างร่างหนังสือตอบกลับที่สอดคล้องกันอย่างชาญฉลาด
- **เทคโนโลยี:** OCR, Information Extraction, Retrieval-Augmented Generation (RAG), Multi-step Chained Prompts

### 🤖 3. ที่ปรึกษาอัจฉริยะ (RAG-Powered Chatbot)
- **หน้าที่:** ผู้ช่วยส่วนตัวที่ทำหน้าที่เป็น "คู่มือมีชีวิต" สามารถตอบคำถามเกี่ยวกับ **วิธีการใช้งานระบบ** และ **ระเบียบงานสารบรรณ** (จากไฟล์ PDF ที่กำหนด)
- **เทคโนโลยี:** Retrieval-Augmented Generation (RAG), Vector Embeddings (`intfloat/multilingual-e5-large`), Query Routing

สำหรับวิดีโอสาธิตการใช้งานโปรเจกต์นี้ [สามารถคลิกดูได้ที่นี่](https://drive.google.com/file/d/1ZJNDUBHFcoi5DBCYQt9cgeA01MqK0xxT/view?usp=sharing)
---

## 🚀 การติดตั้งและใช้งาน (Installation & Setup)

โปรเจกต์นี้ถูกออกแบบมาเพื่อทำงานบน Docker และ Docker Compose เพื่อความสะดวกในการจัดการ Dependencies และการนำไปใช้งาน

### สิ่งที่ต้องมี (Prerequisites)
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- การเชื่อมต่ออินเทอร์เน็ต (สำหรับการตั้งค่าครั้งแรกและการดาวน์โหลดโมเดล)

### ขั้นตอนการติดตั้ง
1.  **Clone a Repository:**
    ```bash
    git clone [[Your-Repository-URL](https://github.com/PK-124960/AI-Innovator25.git)]
    cd [AI-Innovator25]
    ```

2.  **เตรียมไฟล์ระเบียบสารบรรณ:**
    - นำไฟล์ PDF ระเบียบงานสารบรรณของคุณมาวางไว้ในโฟลเดอร์ `k_base/`
    - ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับที่ระบุใน `utils/llm_helper.py` (ค่าเริ่มต้นคือ `k_base/02-sarabun_2566.pdf`)

3.  **ดาวน์โหลด Embedding Model (ทำงานแบบออฟไลน์):**
    - รันสคริปต์เพื่อดาวน์โหลดโมเดลสำหรับทำ Vector Search มาเก็บไว้ในเครื่องก่อนสร้าง Docker image วิธีนี้จะทำให้แอปเริ่มต้นทำงานได้เร็วขึ้นและไม่ขึ้นกับการเชื่อมต่ออินเทอร์เน็ตในภายหลัง
    ```bash
    pip install sentence-transformers
    python download_model.py
    ```
    - รอจนกว่าการดาวน์โหลดจะเสร็จสิ้น จะมีโฟลเดอร์ `models/embedding-model` ปรากฏขึ้นมา

4.  **สร้างและรัน Container ด้วย Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    - คำสั่งนี้จะสร้าง Docker image ที่จำเป็นทั้งหมด (แอป Streamlit, Ollama, Typhoon-OCR) และรันขึ้นมาเป็น service
    - Flag `-d` หมายถึงการรันใน background (detached mode)

5.  **เข้าใช้งานระบบ:**
    - เปิดเว็บเบราว์เซอร์แล้วไปที่: `http://localhost:8501`
    - ระบบพร้อมใช้งาน!

---

## 🏛️ สถาปัตยกรรมของระบบ (System Architecture)

ระบบประกอบด้วย Service หลักที่ทำงานร่วมกันผ่านเครือข่ายของ Docker:

-   **`streamlit_app` (Frontend):**
    -   เป็น Service หลักที่ผู้ใช้โต้ตอบด้วย สร้างจาก Streamlit
    -   ทำหน้าที่เป็น Orchestrator ที่เรียกใช้บริการอื่นๆ
    -   เข้าถึงได้ที่ `http://localhost:8501`

-   **`ollama` (AI Engine):**
    -   Service สำหรับรัน Large Language Models (LLM) และ Embedding Models
    -   ภายในมีการรัน Fine-tuned โมเดล `scb10x/llama3.1-typhoon2-8b-instruct:latest`
    -   แอป Streamlit จะสื่อสารกับ Service นี้ผ่าน HTTP API ที่ `http://ollama:11434`

-   **`typhoon-ocr` (OCR Service):**
    -   Service ที่ทำหน้าที่แปลงรูปภาพเป็นข้อความโดยเฉพาะ
    -   แอป Streamlit จะส่งรูปภาพจากไฟล์ PDF ไปยัง Service นี้ผ่าน HTTP POST Request ที่ `http://typhoon-ocr:8000/process`

---

## 📁 โครงสร้างไฟล์ (File Structure)
```
.
├── assets/                     # เก็บไฟล์รูปภาพสำหรับ UI
├── k_base/                     # เก็บไฟล์ PDF สำหรับ Knowledge Base ของ Chatbot
├── models/                     # เก็บ Embedding Model ที่ดาวน์โหลดแล้ว
├── pages/                      # โค้ดสำหรับแต่ละหน้าในแอป Streamlit
│   ├── 1_✍️_ร่างหนังสือราชการ.py
│   ├── 2_📬_สร้างหนังสือตอบกลับ.py
│   └── 3_🤖_ที่ปรึกษาอัจฉริยะ.py
├── styles/                     # ไฟล์ CSS สำหรับตกแต่ง UI
│   └── main_style.py
├── utils/                      # Helper modules ต่างๆ
│   ├── file_helper.py
│   ├── llm_helper.py           # หัวใจหลักของ Logic AI ทั้งหมด
│   └── ui_helper.py
├── app.py                      # ไฟล์หลักสำหรับหน้าแรก
├── download_model.py           # สคริปต์สำหรับดาวน์โหลดโมเดล
├── docker-compose.yml          # ไฟล์กำหนดการทำงานของ Docker Services
├── Dockerfile                  # คำสั่งสำหรับสร้าง Docker Image ของแอป
└── README.md                   # เอกสารประกอบโปรเจกต์ (ไฟล์นี้)
```
---

## 🔧 การปรับปรุงและพัฒนาต่อ (Future Development)

-   **Vector Database:** เปลี่ยนจากการค้นหาแบบ In-memory ใน `get_relevant_context` ไปใช้ Vector Database จริง เช่น ChromaDB หรือ FAISS เพื่อรองรับ Knowledge Base ขนาดใหญ่และเพิ่มความเร็วในการค้นหา
-   **Fine-tuning:** ทำการ Fine-tune โมเดล LLM ด้วยข้อมูลคู่ "คำสั่ง-ผลลัพธ์" ของเอกสารราชการ เพื่อให้ AI เข้าใจสำนวนและโครงสร้างที่ซับซ้อนได้ดียิ่งขึ้น
-   **Feedback Loop:** พัฒนาระบบ Feedback (`save_feedback`) ให้สมบูรณ์ เพื่อเก็บข้อมูลการแก้ไขของผู้ใช้และนำไปใช้ในการ Fine-tune โมเดลในอนาคต
-   **Authentication:** เพิ่มระบบยืนยันตัวตนสำหรับผู้ใช้งาน
