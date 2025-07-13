# src/retrieval(re-rank).py

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def get_cohere_reranked_retriever(retriever, model_name="rerank-english-v3.0", k=10):
    """
    Wraps a retriever with Cohere Re-Rank for contextual compression.
    """
    compressor = CohereRerank(model=model_name)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

def retrieve_and_answer(question, compression_retriever, llm):
    """
    Retrieves reranked docs and generates an answer using the provided LLM.
    """
    docs = compression_retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"
    )
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    # You must define/provide your own base retriever
    retriever = None  # <-- Replace with your retriever
    compression_retriever = get_cohere_reranked_retriever(retriever)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    question = "How is AI adoption transforming risk management in the financial sector?"
    answer = retrieve_and_answer(question, compression_retriever, llm)
    print(answer)
