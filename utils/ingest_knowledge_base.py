import os
import uuid
import re
import fitz  # PyMuPDF
import qdrant_client

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils.llm_helper import SYSTEM_USAGE_KNOWLEDGE

KNOWLEDGE_BASE_DIR = "k_base"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
COLLECTION_NAME = "rtarf_knowledge_base"  
CHUNK_SIZE_LINES = 15
BATCH_SIZE_EMBEDDING = 32

def clean_text(text: str) -> str:

    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf_file(file_path: str) -> list[dict]:
    
    chunks_with_metadata = []
    filename = os.path.basename(file_path)
    try:
        doc = fitz.open(file_path)
        print(f"📄 Processing '{filename}'...")
        for page_num, page in enumerate(tqdm(doc, desc=f"  -> Pages in {filename}", leave=False)):
            text = page.get_text("text")
            if not text.strip(): continue
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for i in range(0, len(lines), CHUNK_SIZE_LINES):
                chunk_lines = lines[i:i + CHUNK_SIZE_LINES]
                chunk_text = " ".join(chunk_lines)
                cleaned_chunk = clean_text(chunk_text)
                if len(cleaned_chunk) > 50:
                    chunks_with_metadata.append({
                        "text": cleaned_chunk,
                        "source_file": filename,
                        "page_number": page_num + 1
                    })
        doc.close()
        print(f"✅ Finished '{filename}', found {len(chunks_with_metadata)} valid chunks.")
    except Exception as e:
        print(f"❌ Error processing file {filename}: {e}")
    return chunks_with_metadata


def process_system_manual() -> list[dict]:
    
    if not SYSTEM_USAGE_KNOWLEDGE:
        return []
        
    print("⚙️  Processing internal system manual...")
    # แบ่งคู่มือการใช้งานออกเป็นส่วนๆ ตามหัวข้อหลัก (##) ซึ่งเป็นวิธีที่ดีในการ chunking
    chunks = [f"##{chunk}".strip() for chunk in SYSTEM_USAGE_KNOWLEDGE.split('##') if chunk.strip()]
    payloads = []
    for chunk in chunks:
        payloads.append({
            "text": chunk,
            "source_file": "system_manual", # ระบุแหล่งที่มาพิเศษเพื่อให้รู้ว่ามาจากคู่มือ
            "page_number": 1
        })
    print(f"✅ Finished 'system_manual', found {len(payloads)} chunks.")
    return payloads

def initialize_knowledge_base(force_recreate: bool = False):
   
    print("--- Initializing Knowledge Base ---")
    try:
        qdrant_cli = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20)
        # ตรวจสอบการเชื่อมต่อ
        qdrant_cli.get_collections()
        print("✅ Successfully connected to Qdrant.")
    except Exception as e:
        print(f"❌ CRITICAL: Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. Aborting. Error: {e}")
        return

    # ตรวจสอบว่า Collection มีอยู่แล้วหรือไม่
    collection_exists = False
    try:
        qdrant_cli.get_collection(collection_name=COLLECTION_NAME)
        collection_exists = True
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"⚠️ Collection '{COLLECTION_NAME}' not found.")
        collection_exists = False

    # ถ้าบังคับให้สร้างใหม่ หรือ Collection ยังไม่มีอยู่ ให้เริ่มกระบวนการ Ingest
    if force_recreate or not collection_exists:
        if force_recreate:
            print("🔥 Forcing recreation of the knowledge base...")
        else:
            print("🚀 Starting ingestion process for new knowledge base...")

        # --- ส่วนนี้คือ Logic การ Ingest เดิมจากฟังก์ชัน main() ---
        try:
            print("Loading Sentence Transformer model...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            vector_size = embedding_model.get_sentence_embedding_dimension()

            print(f"Recreating Qdrant collection: '{COLLECTION_NAME}'...")
            qdrant_cli.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=vector_size,
                    distance=qdrant_client.http.models.Distance.COSINE
                ),
            )
            
            all_payloads = []
            all_payloads.extend(process_system_manual())
            if os.path.isdir(KNOWLEDGE_BASE_DIR):
                pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith(".pdf")]
                for filename in pdf_files:
                    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    all_payloads.extend(process_pdf_file(file_path))

            if not all_payloads:
                print("❌ No content found to ingest.")
                return

            print(f"Embedding a total of {len(all_payloads)} text chunks...")
            all_texts = [p['text'] for p in all_payloads]
            embeddings = embedding_model.encode(all_texts, show_progress_bar=True, batch_size=BATCH_SIZE_EMBEDDING)

            print("Upserting data into Qdrant...")
            qdrant_cli.upsert(
                collection_name=COLLECTION_NAME,
                points=qdrant_client.http.models.Batch(
                    ids=[str(uuid.uuid4()) for _ in all_payloads],
                    vectors=embeddings.tolist(),
                    payloads=all_payloads
                ),
                wait=True
            )
            print("--- ✅ Knowledge Base Ingestion Process Completed! ---")

        except Exception as e:
            print(f"❌ An error occurred during the ingestion process: {e}")
    else:
        print("--- Knowledge Base is up-to-date. No action needed. ---")

if __name__ == "__main__":
    initialize_knowledge_base(force_recreate=True)