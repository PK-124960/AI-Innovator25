# A Hybrid Intelligence Pipeline for End-to-End Automation of Thai Official Military Correspondence

This project provides a secure, on-premise system to automatically process and reply to official Thai military documents.

## Short Description

This system addresses the challenges of manually handling official correspondence in high-security environments like the Royal Thai Armed Forces (RTARF). It uses a "Hybrid Intelligence Pipeline" to automate the entire workflow, from reading incoming documents to drafting replies. The solution is designed to be secure, efficient, and run on local hardware to prevent data leaks.

## Features / Key Points

*   **End-to-End Automation:** The system manages the full document workflow, framed as a "See, Read, Analyze, and Generate" process.
*   **Secure and On-Premise:** All components run locally within a secure network, avoiding the risks of cloud-based APIs.
    > "To address this gap, this study designs, develops, and evaluates a 'Hybrid Intelligence Pipeline,' an IDP architecture engineered specifically for secure, on-premise operation." 
*   **Complex Document Handling:** It uses a fine-tuned YOLOv8 model for Document Layout Analysis (DLA) to understand the structure of complex military documents.
    > "To tackle complex document structures, a fine-tuned YOLOv8 model is employed to identify the three primary logical regions: Header, Main Body, and Closing..."
*   **High-Accuracy OCR:** A two-stage OCR post-correction process significantly improves text accuracy, especially for military-specific terms.
    > "The combination of a dictionary-based method for predictable, domain-specific errors and a fuzzy matching algorithm for general typographical errors successfully resolves a wide range of OCR inaccuracies." 
*   **Safe Content Generation:** It uses a secure Retrieval-Augmented Generation (RAG) framework with a small, locally-deployed Large Language Model (LLM) to generate accurate and factually grounded replies.
    > "...a Secure Retrieval-Augmented Generation (RAG) framework that utilizes a small, locally-deployable LLM to generate content safely." 
*   **Human-in-the-Loop Verification:** Users verify key steps of the process, which builds trust and ensures the final document matches the user's intent.
    > "This step is critical for building user trust and encouraging system adoption, as it positions the AI as a competent assistant rather than a tool that creates more work."

## Quick Start / Quick Demo

The system provides a web interface for users. The code is available at: `https://github.com/PK-124960/AI-Innovator25.git`.

The general workflow is as follows:
1.  **Login:** The user logs into the system dashboard.
2.  **Upload:** The user uploads a PDF of a "Memorandum" or "Joint News Memo".
3.  **Verify OCR:** The system displays the extracted text. The user can correct any errors.
4.  **Extract Data:** The system automatically extracts key information (like subject, date, sender) into a structured format for user review.
5.  **Confirm Intent:** The system generates three summary options ("Clause 1"). The user selects, edits, and confirms the best one.
6.  **Generate Draft:** The system uses the confirmed information to generate a complete, factually grounded final draft for the user to review, copy, or download.

## Prerequisites

*   **Hardware:** An NVIDIA GeForce RTX 4060 GPU or similar is required to run the models. 
*   **Software:** Docker is used to containerize the application and services. 


## Data

The project uses a dataset of real-world official documents from the Royal Thai Armed Forces.

> "The dataset used for this study consists of real-world official documents from the Royal Thai Armed Forces, comprising two types: 'Memorandums' and 'Joint News Memos.'" 
*   **Fine-tuning Set:** 100 documents (50 of each type).
*   **Test Set:** 20 documents (10 of each type).
  
Key scripts mentioned:
*   `llm_helper.py`
*   `ui_helper.py`

## How it Works / Architecture

The system uses a three-layer architecture, containerized with Docker for easy deployment.

1.  **User Interaction Layer:** A simple web interface built with Streamlit.
2.  **Application & Orchestration Layer:** JupyterLab and Python scripts control the workflow.
3.  **AI & Data Services Layer:** This is the core of the system and includes all on-premise AI models:
    *   **YOLOv8:** For Document Layout Analysis (DLA).
    *   **Typhoon-OCR:** For Optical Character Recognition (OCR).
    *   **Llama-based LLM (Typhoon):** Runs on Ollama for data extraction and content generation.
    *   **Qdrant:** A vector database for the RAG system.

## Tests

Evaluation metrics like Character Accuracy, Word Accuracy, ROUGE-L, and BERTScore were used to validate performance. The document does not provide commands on how to run these tests. (paraphrased from: OCR-result.pdf, p. 30, Section 3.6)

## Maintainers / Contact

*   **Author:** Ponkrit Kaewsawee

## Acknowledgements

> "I am deeply grateful to the AIT Scholarship Committee for their generous support... I would like to express my deepest gratitude to my supervisor, Prof. Chaklam Silpasuwanchai, for his invaluable guidance, mentorship, and unwavering support... My sincere thanks also go to the TA, Mr. Akaradet, for his time and constructive comments. I am also grateful to the Royal Thai Armed Forces for providing the opportunity and resources that made this real-world study possible." 
