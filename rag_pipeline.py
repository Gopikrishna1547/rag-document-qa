import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def load_pdf(filepath):
    log.info(f"Loading PDF: {filepath}")
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    log.info(f"Loaded {len(pages)} pages")
    return pages


def split_documents(pages):
    log.info("Splitting into chunks...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    log.info(f"Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    log.info("Creating FAISS vector store...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    log.info("Vector store ready!")
    return vector_store


def retrieve_context(vector_store, query, k=3):
    log.info("Retrieving relevant chunks...")
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def generate_answer(context, question):
    log.info("Generating answer...")
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch

    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

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
    return answer


def main():
    print("\n" + "="*50)
    print("  RAG-Based Document Q&A System")
    print("  Powered by LangChain + FAISS + Hugging Face")
    print("="*50 + "\n")

    pdf_files = list(Path("data").glob("*.pdf"))
    if not pdf_files:
        print("No PDF found in data/ folder!")
        print("Add a PDF file to the data/ folder and run again.")
        return

    pdf_path = str(pdf_files[0])
    print(f"Using PDF: {pdf_path}\n")

    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)
    vector_store = create_vector_store(chunks)

    print("\nReady! Ask any question about the document.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            context = retrieve_context(vector_store, question)
            answer = generate_answer(context, question)
            print(f"\nAnswer: {answer}\n")
            print("-" * 50)
        except Exception as e:
            log.error(f"Error: {e}")
            print(f"Could not generate answer: {e}\n")


if __name__ == "__main__":
    main()
