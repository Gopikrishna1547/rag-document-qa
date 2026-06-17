# RAG-Based Document Q&A System

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about any PDF document — powered entirely by free, open-source models (no API key needed!).

## How It Works
```
PDF Document
     ↓
Split into chunks (LangChain)
     ↓
Convert to vectors (Hugging Face Embeddings)
     ↓
Store in FAISS vector database
     ↓
User asks a question
     ↓
Find most relevant chunks (similarity search)
     ↓
Generate answer (Hugging Face QA model)
     ↓
Display answer to user
```

## Project Structure
```
rag-document-qa/
├── rag_pipeline.py     # Main RAG pipeline
├── data/               # Put your PDF files here
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Gopikrishna1547/rag-document-qa.git
cd rag-document-qa
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your PDF**

Copy any PDF file into the `data/` folder.

**4. Run**
```bash
python3 rag_pipeline.py
```

**5. Ask questions!**
```
❓ Your question: What is this document about?
💡 Answer: This document is about...
```

## Tech Stack
- Python 3.10+
- LangChain — document loading and text splitting
- FAISS — vector similarity search
- Hugging Face Sentence Transformers — text embeddings
- Hugging Face Transformers — question answering model
- PyPDF — PDF reading

## Author
Gopikrishna Bojedla
[LinkedIn](https://www.linkedin.com/in/gopi-krishna-83856320a)
