# Multimodal RAG System

*A Retrieval-Augmented Generation app that understands PDFs, Images, and Audio.*

![LangChain](https://img.shields.io/badge/LangChain-0A0A0A?logo=langchain\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit\&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00599C?logo=facebook\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3670A0?logo=python\&logoColor=ffdd54)

---

## Overview

The **Multimodal RAG System** is an end-to-end application that allows users to upload **PDFs, images, and audio files** and interact with their contents through natural language.
It combines **data extraction**, **semantic retrieval**, and **context-aware generation** to deliver grounded answers from multimodal documents.

---

## Features

* **Document Ingestion:** Upload and process PDFs, images, and audio files.
* **Content Extraction:**
  * Text and tables from PDFs
  * OCR and BLIP for images
  * Speech-to-text transcription for audio
* **Semantic Retrieval:** Vector embeddings stored and queried via **FAISS**.
* **Context-Aware QA:** Uses **Llama-3.8B** through **LangChain** for accurate, context-grounded responses.
* **Interactive UI:** Built with **Streamlit** for an intuitive chat interface.

---

## Tech Stack

| Component          | Tools / Libraries                                                    |
| ------------------ | -------------------------------------------------------------------- |
| **Language Model** | Llama-3.8B                                                           |
| **Framework**      | LangChain                                                            |
| **Frontend / UI**  | Streamlit                                                            |
| **Vector Store**   | FAISS                                                                |
| **Processing**     | PyPDF2 / pdfplumber, pytesseract, OpenAI Whisper / SpeechRecognition |
| **Language**       | Python 3.x                                                           |

---

## How It Works

1. **Upload Files** → PDFs, images, or audio.
2. **Extract Data** → Text, tables, and transcriptions are extracted and chunked.
3. **Embed & Index** → Each chunk is embedded and stored in FAISS for semantic retrieval.
4. **Ask Questions** → User queries are matched against relevant chunks.
5. **Generate Answers** → Llama-3.8B generates answers with retrieved context.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/AryakumarMishra/Multimodal-RAG-Chatbot
cd multimodal-rag

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Project Structure (Major files)

```
multimodal-rag/
│
├── app.py                  # Streamlit UI
├── processing.py           # Data extraction pipelines (PDF, Image, Audio)
├── experiment.ipynb        # Jupyter notebook for writing and testing codes
└── requirements.txt
```

---

## Future Improvements

* Support for **video** ingestion and frame-level retrieval
* Add **reranker** for improved retrieval precision
* Deploy backend via **FastAPI** and **Docker**

---
