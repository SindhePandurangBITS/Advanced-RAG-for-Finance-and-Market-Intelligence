# src/indexing(vectorize_chroma).py

from langchain_community.vectorstores import Chroma

def build_chroma_vectorstore(all_texts, embd):
    """
    Build and return a Chroma vectorstore retriever.
    Args:
        all_texts (list): List of text chunks/summaries.
        embd: Embedding model (e.g. OpenAIEmbeddings()).
    Returns:
        retriever: LangChain-compatible retriever.
    """
    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
    return vectorstore.as_retriever()

if __name__ == "__main__":
    # Example usage: assumes all_texts and embd already defined
    all_texts = []  # Provide your list of text chunks/summaries
    embd = None     # Provide your embedding model instance
    retriever = build_chroma_vectorstore(all_texts, embd)

    # Example: RAG generation using this retriever
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI

    prompt = hub.pull("rlm/rag-prompt")
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = "How is AI adoption transforming fraud detection and risk management in the financial sector?"
    print(rag_chain.invoke(question))
