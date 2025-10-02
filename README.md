# A Hybrid Intelligence Pipeline for End-to-End Automation of Thai Official Military Correspondence

This project provides a secure, on-premise system to automatically process and reply to official Thai military documents.

## Short Description

This system addresses the challenges of manually handling official correspondence in high-security environments like the Royal Thai Armed Forces (RTARF). It uses a "Hybrid Intelligence Pipeline" to automate the entire workflow, from reading incoming documents to drafting replies. The solution is designed to be secure, efficient, and run on local hardware to prevent data leaks.

## Features / Key Points

*   **End-to-End Automation:** The system manages the full document workflow, framed as a "See, Read, Analyze, and Generate" process.
*   **Secure and On-Premise:** All components run locally within a secure network, avoiding the risks of cloud-based APIs.
    > "To address this gap, this study designs, develops, and evaluates a 'Hybrid Intelligence Pipeline,' an IDP architecture engineered specifically for secure, on-premise operation." (source: OCR-result.pdf, p. 4, Abstract)
*   **Complex Document Handling:** It uses a fine-tuned YOLOv8 model for Document Layout Analysis (DLA) to understand the structure of complex military documents.
    > "To tackle complex document structures, a fine-tuned YOLOv8 model is employed to identify the three primary logical regions: Header, Main Body, and Closing..." (source: OCR-result.pdf, p. 22, Section 3.1.1)
*   **High-Accuracy OCR:** A two-stage OCR post-correction process significantly improves text accuracy, especially for military-specific terms.
    > "The combination of a dictionary-based method for predictable, domain-specific errors and a fuzzy matching algorithm for general typographical errors successfully resolves a wide range of OCR inaccuracies." (source: OCR-result.pdf, p. 33, Section 4.1.1)
*   **Safe Content Generation:** It uses a secure Retrieval-Augmented Generation (RAG) framework with a small, locally-deployed Large Language Model (LLM) to generate accurate and factually grounded replies.
    > "...a Secure Retrieval-Augmented Generation (RAG) framework that utilizes a small, locally-deployable LLM to generate content safely." (source: OCR-result.pdf, p. 4, Abstract)
*   **Human-in-the-Loop Verification:** Users verify key steps of the process, which builds trust and ensures the final document matches the user's intent.
    > "This step is critical for building user trust and encouraging system adoption, as it positions the AI as a competent assistant rather than a tool that creates more work." (source: OCR-result.pdf, p. 35, Section 4.2.3)

## Quick Start / Quick Demo

The system provides a web interface for users. The code is available at: `https://github.com/PK-124960/AI-Innovator25.git` (source: OCR-result.pdf, p. 32, Figure 4.1 Caption).

The general workflow is as follows (paraphrased from: OCR-result.pdf, p. 40, Appendix A.2):
1.  **Login:** The user logs into the system dashboard.
2.  **Upload:** The user uploads a PDF of a "Memorandum" or "Joint News Memo".
3.  **Verify OCR:** The system displays the extracted text. The user can correct any errors.
4.  **Extract Data:** The system automatically extracts key information (like subject, date, sender) into a structured format for user review.
5.  **Confirm Intent:** The system generates three summary options ("Clause 1"). The user selects, edits, and confirms the best one.
6.  **Generate Draft:** The system uses the confirmed information to generate a complete, factually grounded final draft for the user to review, copy, or download.

## Prerequisites

*   **Hardware:** An NVIDIA GeForce RTX 4060 GPU or similar is required to run the models. (source: OCR-result.pdf, p. 29, Section 3.4)
*   **Software:** Docker is used to containerize the application and services. (source: OCR-result.pdf, p. 21, Section 3)
*   `[[MISSING - a full list of software prerequisites, including Python version]]`

## Installation

# [[MISSING - installation commands]]
# 1. Clone the repository
# git clone https://github.com/PK-124960/AI-Innovator25.git
#
# [[MISSING - steps to build and run docker containers]]

## Usage Examples

`[[MISSING - example code blocks or specific usage commands]]`

## Data

The project uses a dataset of real-world official documents from the Royal Thai Armed Forces.

> "The dataset used for this study consists of real-world official documents from the Royal Thai Armed Forces, comprising two types: 'Memorandums' and 'Joint News Memos.'" (source: OCR-result.pdf, p. 29, Section 3.5)
*   **Fine-tuning Set:** 100 documents (50 of each type).
*   **Test Set:** 20 documents (10 of each type).

## File / Folder Structure

`[[MISSING - a tree or detailed list of the project's file structure]]`

Key scripts mentioned:
*   `llm_helper.py`
*   `ui_helper.py`
(source: OCR-result.pdf, p. 21, Figure 3.1)

## How it Works / Architecture

The system uses a three-layer architecture, containerized with Docker for easy deployment.

1.  **User Interaction Layer:** A simple web interface built with Streamlit.
2.  **Application & Orchestration Layer:** JupyterLab and Python scripts control the workflow.
3.  **AI & Data Services Layer:** This is the core of the system and includes all on-premise AI models:
    *   **YOLOv8:** For Document Layout Analysis (DLA).
    *   **Typhoon-OCR:** For Optical Character Recognition (OCR).
    *   **Llama-based LLM (Typhoon):** Runs on Ollama for data extraction and content generation.
    *   **Qdrant:** A vector database for the RAG system.

(paraphrased from: OCR-result.pdf, p. 21, Chapter 3)

## Configuration

`[[MISSING - details on environment variables or configuration files]]`

## Tests

Evaluation metrics like Character Accuracy, Word Accuracy, ROUGE-L, and BERTScore were used to validate performance. The document does not provide commands on how to run these tests. (paraphrased from: OCR-result.pdf, p. 30, Section 3.6)

## Contributing

`[[This is a generic template, as no contribution guide was provided in the source document]]`

We welcome contributions! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

`[[MISSING - license info]]`

## Maintainers / Contact

*   **Author:** Ponkrit Kaewsawee
*   `[[MISSING - maintainer email or contact information]]`

## Acknowledgements

> "I am deeply grateful to the AIT Scholarship Committee for their generous support... I would like to express my deepest gratitude to my supervisor, Prof. Chaklam Silpasuwanchai, for his invaluable guidance, mentorship, and unwavering support... My sincere thanks also go to the TA, Mr. Akaradet, for his time and constructive comments. I am also grateful to the Royal Thai Armed Forces for providing the opportunity and resources that made this real-world study possible." (source: OCR-result.pdf, p. iii, Acknowledgements)

## Troubleshooting / FAQ

`[[MISSING - troubleshooting guide or frequently asked questions]]`

## References / Sources

*   **Source Document:** `OCR-result.pdf` - A special study paper titled "A Hybrid Intelligence Pipeline for End-to-End Automation of Thai Official Military Correspondence". This document was the primary source for all information in this README.
    *   Project goal and description from the Abstract and Chapter 1.
    *   Architecture and technical details from Chapter 3 and Figures 3.1, 3.2.
    *   Dataset information from Section 3.5.
    *   User workflow from Appendix A.2.
    *   Acknowledgements from page iii.
