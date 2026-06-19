import os
import logging
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import tempfile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Page config ──
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon=" ",
    layout="centered"
)

st.title(" RAG Document Q&A System")
st.markdown("Upload any PDF and ask questions about it!")
st.divider()

# ── Load model once (cached) ──
@st.cache_resource
def load_model():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ── Process uploaded PDF ──
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.unlink(tmp_path)
    return vector_store, len(pages), len(chunks)


# ── Answer question ──
def answer_question(vector_store, question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    tokenizer, model = load_model()
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(
        inputs["input_ids"][0][start:end],
        skip_special_tokens=True
    )
    return answer, context


# ── UI ──
uploaded_file = st.file_uploader(
    "Upload your PDF here",
    type=["pdf"],
    help="Upload any PDF document"
)

if uploaded_file:
    with st.spinner("Reading and processing your PDF..."):
        vector_store, num_pages, num_chunks = process_pdf(uploaded_file)

    st.success(f" PDF loaded! {num_pages} pages, {num_chunks} chunks created.")
    st.divider()

    st.subheader("Ask a question about your document")
    question = st.text_input(
        "Your question:",
        placeholder="e.g. What are the main topics? What is the conclusion?"
    )

    if st.button("Get Answer", type="primary"):
        if question.strip():
            with st.spinner("Finding answer..."):
                answer, context = answer_question(vector_store, question)

            st.subheader("Answer:")
            st.success(answer if answer.strip() else "Could not find a specific answer. Try rephrasing your question.")

            with st.expander("View relevant context from PDF"):
                st.write(context)
        else:
            st.warning("Please type a question first!")

else:
    st.info(" Upload a PDF file above to get started!")

    st.markdown("### Example questions you can ask:")
    st.markdown("""
    - What is this document about?
    - What are the main points?
    - When was iPhone 16 launched?
    - Who created Python?
    - What are the key findings?
    """)
