from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = 'intfloat/multilingual-e5-large'
MODEL_PATH = './models/embedding-model' 

print(f"Downloading model '{MODEL_NAME}' to '{MODEL_PATH}'...")

# check if directory don't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

model = SentenceTransformer(MODEL_NAME)
model.save(MODEL_PATH)

print("Model downloaded and saved successfully!")