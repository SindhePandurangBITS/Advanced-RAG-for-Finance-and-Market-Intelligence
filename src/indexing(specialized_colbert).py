# src/indexing(specialized_colbert).py

from ragatouille import RAGPretrainedModel

def build_colbert_index(all_texts, index_name="my_rag_index"):
    """
    Build and return a ColBERT RAGPretrainedModel retriever.
    Args:
        all_texts (list): List of strings (chunks/summaries).
        index_name (str): Name for the ColBERT index.
    Returns:
        retriever: LangChain-compatible retriever.
    """
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG.index(
        collection=all_texts,
        index_name=index_name,
        max_document_length=180,
        split_documents=True,
    )
    retriever = RAG.as_langchain_retriever(k=3)
    return retriever

if __name__ == "__main__":
    # Example usage: assumes all_texts is already defined
    all_texts = []  # Fill in with your processed chunk list
    retriever = build_colbert_index(all_texts)
    # Example query
    query = "How is AI adoption transforming risk management in the financial sector?"
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:\n{doc.page_content[:400]}\n")
