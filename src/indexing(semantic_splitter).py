# src/Indexing(Semantic_Splitter).py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def semantic_split_docs(all_docs, chunk_size=800, chunk_overlap=100):
    """
    Splits documents into semantic chunks.
    Args:
        all_docs (list): List of LangChain Document objects.
        chunk_size (int): Size of each chunk (tokens/characters).
        chunk_overlap (int): Overlap between chunks.
    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    leaf_texts = []
    for doc in all_docs:
        splits = text_splitter.split_text(doc.page_content)
        leaf_texts.extend(splits)  # Only the split text, not metadata
    return leaf_texts

if __name__ == "__main__":
    # Example usage: assumes `all_docs` already exists (loaded from utils)
    # from utils.download_and_load_pdfs import download_and_load_pdfs
    # all_docs = download_and_load_pdfs(pdf_urls)
    
    # For demo, replace all_docs with your data loader result
    all_docs = []  # Load or import your docs list here
    leaf_texts = semantic_split_docs(all_docs)
    print(f"Total chunks: {len(leaf_texts)}")
    
    # Embedding model (set up for downstream indexing)
    embd = OpenAIEmbeddings()
    print("Embedding model is ready.")
